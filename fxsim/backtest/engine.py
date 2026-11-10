"""バックテストのエンジン(ポジションは常に 1 つまで、ネット)。

1 本の足 i で起きる順番:
    1. ロールオーバーをまたいだ分のスワップ
    2. 前の足の終わりに出した注文を、足 i の始値で約定(買いは ask、売りは bid)。ラストルックで拒否されることがある
    3. 足 i の中で損切り・利確・ロスカットを判定。broker.intrabar="path" なら足の中の経路で最初に越えた点で約定
       (経路の無い足と "pessimistic" は、足の高安で判定し、両方届いたら損切りを先にする)
    4. 終値で評価損益
    5. strategy.on_bar(i) を呼ぶ(ここで出した注文は次の足の始値で約定する)
費用(スプレッド・スリッページ・手数料・スワップ)は取引ごとに記録する。
戦略は足 i の終値までしか見られないので、先読みは構造的に起きない
(ただし prepare で未来を使う指標を作らないこと。backtest.lookahead で検査できる)。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ..calendar import rollover_weight, trading_day
from ..data import Bars, load_rates
from .broker import BrokerConfig
from .schedule import TradingSchedule


@dataclass
class EngineConfig:
    initial_balance: float = 1_000_000.0   # 円
    sizing: str = "fixed"                  # fixed | risk
    units: float = 10_000.0                # fixed のときの数量
    risk_fraction: float = 0.01            # risk のとき、損切りまでの損失 = 有効証拠金 x これ
    min_units: float = 1_000.0
    max_units: float = 1_000_000.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    side: int            # +1 買い / -1 売り
    units: float
    entry_price: float
    entry_mid: float
    entry_idx: int
    sl: float | None
    tp: float | None
    swap: float = 0.0
    commission: float = 0.0       # 円(建てたときの分。決済の分は決済時に足す)
    slippage: float = 0.0         # 円(不利な向きを正)


@dataclass
class Trade:
    side: int
    units: float
    entry_t: int
    exit_t: int
    entry_price: float
    exit_price: float
    pnl: float            # 円(スワップ・手数料込み)
    pips: float           # 約定価格どうしの pip(スプレッド・スリッページ込み)
    gross_pips: float     # mid どうしの pip(コスト抜き = 戦略の素の優位性)
    cost_pips: float      # gross - pips(スプレッド + スリッページ)
    swap: float           # 円
    reason: str           # signal | sl | tp | hunted | losscut | end | reverse
    bars_held: int
    commission: float = 0.0   # 円(往復)
    slippage: float = 0.0     # 円(往復、不利な向きを正)


@dataclass
class _Order:
    kind: str                # enter | close
    side: int = 0
    sl_dist: float | None = None
    tp_dist: float | None = None
    units: float | None = None


class StrategyContext:
    """戦略から見えるもの。注文は次の足の始値で約定する。"""

    def __init__(self, engine: Backtester):
        self._e = engine

    # ---- 参照 ----
    @property
    def position(self) -> Position | None:
        return self._e.pos

    @property
    def side(self) -> int:
        return self._e.pos.side if self._e.pos else 0

    @property
    def equity(self) -> float:
        return self._e.equity

    @property
    def balance(self) -> float:
        return self._e.balance

    @property
    def pip(self) -> float:
        return self._e.pip

    @property
    def bars(self) -> Bars:
        return self._e.bars

    @property
    def rng(self) -> np.random.Generator:
        return self._e.strategy_rng

    # ---- 注文 ----
    @property
    def can_enter(self) -> bool:
        """いま新規注文を出せるか(動作ケースの取引時間)。"""
        return bool(self._e.can_enter[self._e.i])

    def buy(self, sl: float | None = None, tp: float | None = None, units: float | None = None) -> None:
        """買い。sl / tp は約定価格からの距離(価格単位)。売り持ちなら決済してから買う(ドテン)。
        取引時間外なら新規は出さない(売り持ちの決済だけ行う)。"""
        self._enter(1, sl, tp, units)

    def sell(self, sl: float | None = None, tp: float | None = None, units: float | None = None) -> None:
        self._enter(-1, sl, tp, units)

    def _enter(self, side: int, sl, tp, units) -> None:
        e = self._e
        if e.can_enter[e.i]:
            e._submit(_Order("enter", side, sl, tp, units))
        elif e.pos is not None and e.pos.side == -side:
            e._submit(_Order("close"))

    def close(self) -> None:
        if self._e.pos is not None:
            self._e._submit(_Order("close"))

    def set_sl(self, price: float | None) -> None:
        """損切り価格(絶対値)を変える。次の足から有効。"""
        if self._e.pos is not None:
            self._e.pos.sl = price

    def set_tp(self, price: float | None) -> None:
        if self._e.pos is not None:
            self._e.pos.tp = price


ProgressFn = Callable[["Backtester", int], None]


def _swap_days(t: np.ndarray) -> list[int]:
    """足 i の直前(足 i-1 との間)に付くスワップの日数。取引日の切り替わりごとに rollover_weight を足す。"""
    day = trading_day(t).astype(np.int64)
    out = [0] * len(t)
    for i in np.flatnonzero(np.diff(day) > 0) + 1:
        out[i] = sum(rollover_weight(d) for d in range(int(day[i - 1]), int(day[i])))
    return out


def _first_cross(values: np.ndarray, level: float, below: bool) -> int:
    """values が level 以下(below)/ 以上 になる最初の位置。無ければ -1。"""
    hit = values <= level if below else values >= level
    j = int(np.argmax(hit))
    return j if hit[j] else -1


class Backtester:
    def __init__(self, bars: Bars, strategy, broker: BrokerConfig | None = None,
                 config: EngineConfig | None = None, pip: float = 0.01, seed: int = 0,
                 schedule: TradingSchedule | None = None):
        self.bars = bars
        self.schedule = schedule or TradingSchedule()
        self.decide, self.can_enter, self.force_close = self.schedule.masks(bars.t, bars.timeframe)
        # 足ごとの「直前の足からのロールオーバーで付くスワップの日数」(水曜は 3 日分、週末は 0)
        self.swap_days = _swap_days(bars.t)
        self.strategy = strategy
        self.broker = broker or BrokerConfig()
        self.cfg = config or EngineConfig()
        self.pip = pip
        self.rng = np.random.default_rng([seed, 7919])          # ブローカー用
        self.strategy_rng = np.random.default_rng([seed, 104729])
        self.balance = self.cfg.initial_balance
        self.equity = self.balance
        self.pos: Position | None = None
        self.pending: list[_Order] = []
        self.trades: list[Trade] = []
        self.equity_curve = np.full(len(bars), np.nan)
        self.balance_curve = np.full(len(bars), np.nan)
        self.decisions: list[tuple[int, str, int]] = []   # 先読み検査用 (i, kind, side)
        self.i = 0
        self.ruined = False
        self.stats = {"hunted": 0, "losscut": 0, "swap_total": 0.0, "slippage_pips": 0.0, "commission_total": 0.0,
                      "slippage_total": 0.0, "rejected": 0, "zero_cut": 0.0}
        self._prepare_prices()

    # ---- 足ごとの価格(ブローカーの上乗せ込み)を先に用意する ----
    def _prepare_prices(self) -> None:
        b, bars, pip = self.broker, self.bars, self.pip
        markup, floor = b.spread_markup_pips * pip, b.spread_floor_pips * pip

        def widen(s):
            """ブローカーの上乗せと下限を掛けたスプレッドと、市場のスプレッドとの差の半分。"""
            return (np.maximum(s + markup, floor) - s) / 2

        s_open = 2 * (bars.open - bars.bid_open)
        s_close = 2 * (bars.close - bars.bid_close)
        x_open, x_close = widen(s_open), widen(s_close)
        x_hl = max(markup, floor) / 2           # 足の中の高安には上限で見積もる(経路で確かめる)
        self._widen = widen
        self.o, self.c = bars.open.tolist(), bars.close.tolist()
        self.bo = (bars.bid_open - x_open).tolist()
        self.ao = (bars.ask_open + x_open).tolist()
        self.bc = (bars.bid_close - x_close).tolist()
        self.ac = (bars.ask_close + x_close).tolist()
        self.bl = (bars.bid_low - x_hl).tolist()
        self.bh = bars.bid_high.tolist()
        self.al = bars.ask_low.tolist()
        self.ah = (bars.ask_high + x_hl).tolist()
        self.sm = (np.maximum(bars.spread_mean + markup, floor)).tolist()
        self.t = bars.t.tolist()
        self.use_path = b.intrabar == "path" and bars.base_mid is not None
        if b.swap_mode == "rates":
            self.rate_diff = load_rates(b.rates_path).differential(bars.t).tolist()

    # ---- 注文の受付 ----
    def _submit(self, order: _Order) -> None:
        self.decisions.append((self.i, order.kind, order.side))
        if order.kind == "enter":
            self.pending = [o for o in self.pending if o.kind != "enter"]
        self.pending.append(order)

    # ---- 約定まわり ----
    def _slip(self, i: int, stop: bool = False) -> float:
        b = self.broker
        size = b.slippage_pips + b.slippage_spread_k * self.sm[i] / self.pip
        if size <= 0:
            return 0.0
        z = self.rng.standard_normal()
        if b.adverse_only:
            z = abs(z)
        s = z * size * (b.stop_slippage_mult if stop else 1.0)
        self.stats["slippage_pips"] += s
        return s * self.pip   # 正 = 不利

    def _commission(self, units: float) -> float:
        fee = self.broker.commission_per_10k * units / 10_000
        self.balance -= fee
        self.stats["commission_total"] += fee
        return fee

    def _units(self, sl_dist: float | None, price: float, units: float | None) -> float:
        c = self.cfg
        if units is None:
            if c.sizing == "risk" and sl_dist:
                units = self.equity * c.risk_fraction / sl_dist
            else:
                units = c.units
        cap = max(0.0, self.equity) * self.broker.leverage / price
        units = min(units, cap, c.max_units)
        return float(np.floor(units / 1000.0) * 1000.0)

    def _last_look_rejects(self, i: int, side: int) -> bool:
        """判断した時点(前の足の終値)より客に有利な始値なら、確率で拒否する。"""
        p = self.broker.last_look_prob
        if p <= 0 or i == 0:
            return False
        favorable = (self.o[i] < self.c[i - 1]) if side > 0 else (self.o[i] > self.c[i - 1])
        if favorable and self.rng.random() < p:
            self.stats["rejected"] += 1
            return True
        return False

    def _open(self, i: int, o: _Order) -> None:
        slip = self._slip(i)
        price = self.ao[i] + slip if o.side > 0 else self.bo[i] - slip
        units = self._units(o.sl_dist, price, o.units)
        if units < self.cfg.min_units:
            return
        sl = price - o.side * o.sl_dist if o.sl_dist else None
        tp = price + o.side * o.tp_dist if o.tp_dist else None
        fee = self._commission(units)
        self.stats["slippage_total"] += slip * units
        self.pos = Position(o.side, units, price, self.o[i], i, sl, tp, commission=fee, slippage=slip * units)

    def _close(self, i: int, price: float, mid: float, reason: str, slip: float = 0.0) -> None:
        p = self.pos
        fee = self._commission(p.units)
        self.stats["slippage_total"] += slip * p.units
        pips = (price - p.entry_price) * p.side / self.pip
        gross = (mid - p.entry_mid) * p.side / self.pip
        commission = p.commission + fee
        pnl = (price - p.entry_price) * p.side * p.units + p.swap - commission
        self.balance += (price - p.entry_price) * p.side * p.units   # スワップと手数料は発生時に残高へ入れている
        self.trades.append(Trade(p.side, p.units, int(self.t[p.entry_idx]), int(self.t[i]), p.entry_price, price,
                                 pnl, pips, gross, gross - pips, p.swap, reason, i - p.entry_idx,
                                 commission, p.slippage + slip * p.units))
        self.pos = None
        if self.balance < 0 and self.broker.zero_cut:          # ゼロカット: マイナス残高はブローカー持ち
            self.stats["zero_cut"] += -self.balance
            self.balance = 0.0

    def _close_at_open(self, i: int, reason: str) -> None:
        slip = self._slip(i)
        price = self.bo[i] - slip if self.pos.side > 0 else self.ao[i] + slip
        self._close(i, price, self.o[i], reason, slip)

    def _path_quotes(self, i: int, side: int):
        """足 i の中の経路で、建玉を閉じる側の価格(買いなら bid、売りなら ask)と mid。経路が無ければ None。

        先頭は 1 つ前の点(越えた 1 歩が飛びかどうかを見るため)。
        """
        if not self.use_path:
            return None
        path = self.bars.path(i, prev=True)
        if path is None or len(path[0]) < 2:
            return None
        mid, spread = path
        half = (spread + 2 * self._widen(spread)) / 2
        return (mid - half if side > 0 else mid + half), mid

    def _check_stops(self, i: int) -> None:
        p = self.pos
        sd = p.side
        # 足の高安で、損切り・利確・ロスカットに届いた可能性があるかを先に見る
        lo = self.bl[i] if sd > 0 else -self.ah[i]           # 売りは符号を反転して買いと同じ式にする
        hi = self.bh[i] if sd > 0 else -self.al[i]
        sl = p.sl * sd if p.sl is not None else None
        tp = p.tp * sd if p.tp is not None else None
        lc = self._losscut_price(i)
        lc = lc * sd if lc is not None else None
        stop = max(x for x in (sl, lc, -np.inf) if x is not None)   # 先に届く方(高い方)が実質の逆指値
        touch_stop = stop > -np.inf and lo <= stop
        touch_tp = tp is not None and hi >= tp
        hunted = False
        if (sl is not None and not touch_stop and self.broker.hunt_prob > 0
                and lo - sl <= self.broker.hunt_pips * self.pip and self.rng.random() < self.broker.hunt_prob):
            hunted = True
            self.stats["hunted"] += 1
        if not (touch_stop or touch_tp or hunted):
            return
        quotes = self._path_quotes(i, sd) if not hunted else None
        if quotes is not None:
            q, mid = quotes
            q = q * sd
            js = _first_cross(q[1:], stop, below=True) + 1 if touch_stop else 0     # q[0] は 1 つ前の点
            jt = _first_cross(q[1:], tp, below=False) + 1 if touch_tp else 0
            if js <= 0 and jt <= 0:
                return
            # 越えた 1 歩が飛びのときだけ、飛んだ先で約定する。ふつうの 1 歩なら点の間も連続して動いている。
            # 足の中の点が少ない(日足の再生など)と「ふつうの 1 歩」が測れないので、越えた点で約定する(悲観的)
            if len(mid) > 10:
                gap = self.broker.gap_k * max(float(np.median(np.abs(np.diff(mid)))), 1e-12)
            else:
                gap = 0.0
            if js > 0 and (jt <= 0 or js <= jt):
                slip = self._slip(i, stop=True)
                jumped = q[js - 1] - q[js] > gap
                fill = (min(q[js], stop) if jumped else stop) - slip
                reason = "losscut" if lc is not None and stop == lc and (sl is None or lc > sl) else "sl"
                if reason == "losscut":
                    self.stats["losscut"] += 1
                at = float(mid[js]) if jumped else float(stop * sd + (mid[js] - q[js] * sd))   # 損切り価格に届いた瞬間
                self._close(i, fill * sd, at, reason, slip)
            else:
                jumped = q[jt] - q[jt - 1] > gap
                fill = q[jt] if (self.broker.positive_slippage and jumped and q[jt] > tp) else tp
                # 素の損益の mid: 飛びなら越えた点の mid、ふつうの 1 歩なら利確価格に届いた瞬間の mid
                at = float(mid[jt]) if jumped else float(tp * sd + (mid[jt] - q[jt] * sd))
                self._close(i, fill * sd, at, "tp")
            return
        # 経路が無い足: 足の高安だけで判定(両方届いたら損切りが先 = 悲観的)
        half = self.sm[i] / 2
        o_ = self.bo[i] if sd > 0 else -self.ao[i]
        if touch_stop and o_ <= stop:
            slip = self._slip(i, stop=True)
            fill = o_ - slip
            self._close(i, fill * sd, fill * sd + sd * half, "sl", slip)
        elif tp is not None and o_ >= tp:
            fill = o_ if self.broker.positive_slippage else tp
            self._close(i, fill * sd, fill * sd + sd * half, "tp")
        elif touch_stop or hunted:
            slip = self._slip(i, stop=True)
            fill = stop - slip if touch_stop else sl - slip
            level = stop if touch_stop else sl
            reason = "hunted" if hunted else ("losscut" if lc is not None and stop == lc and (sl is None or lc > sl)
                                              else "sl")
            if reason == "losscut":
                self.stats["losscut"] += 1
            self._close(i, fill * sd, (level + half) * sd, reason, slip)
        elif touch_tp:
            self._close(i, tp * sd, (tp + half) * sd, "tp")

    def _losscut_price(self, i: int) -> float | None:
        """有効証拠金が「必要証拠金 x losscut_level」を割る価格(必要証拠金は前の足の終値で見積もる)。"""
        p = self.pos
        level = self.broker.losscut_level
        if level <= 0:
            return None
        ref = self.c[i - 1] if i > 0 else self.o[i]
        margin = p.units * ref / self.broker.leverage * level
        return p.entry_price - p.side * (self.balance - margin) / p.units

    def _swap(self, i: int) -> None:
        if self.pos is None:
            return
        days = self.swap_days[i]
        if not days:
            return
        b, p = self.broker, self.pos
        if b.swap_mode == "rates":
            # 買いは金利差を受け取り、売りは払う。ブローカーは両側から swap_markup を取る(年率)
            annual = p.side * self.rate_diff[i] - b.swap_markup
            amt = p.units * self.c[i - 1] * annual * days / 365
        else:
            rate = b.swap_long_per_10k if p.side > 0 else b.swap_short_per_10k
            amt = rate * days * p.units / 10_000
        p.swap += amt
        self.balance += amt
        self.stats["swap_total"] += amt

    def _mark(self, i: int) -> None:
        p = self.pos
        if p is None:
            self.equity = self.balance
            return
        px = self.bc[i] if p.side > 0 else self.ac[i]
        self.equity = self.balance + (px - p.entry_price) * p.side * p.units
        margin = p.units * self.c[i] / self.broker.leverage
        if self.equity < margin * self.broker.losscut_level:     # 経路で拾えなかったとき(経路の無い足など)の保険
            self._close(i, px, self.c[i], "losscut")
            self.stats["losscut"] += 1
            self.equity = self.balance

    # ---- 実行 ----
    def run(self, progress: ProgressFn | None = None, every: int = 500) -> Backtester:
        n = len(self.bars)
        self.strategy.prepare(self.bars, self.strategy_rng)
        ctx = StrategyContext(self)
        warmup = int(getattr(self.strategy, "warmup", 0))
        for i in range(n):
            self.i = i
            if self.ruined:
                self.equity_curve[i] = self.equity
                self.balance_curve[i] = self.balance
                continue
            self._swap(i)
            if self.pending:
                orders, self.pending = self.pending, []
                for o in orders:
                    if o.kind == "close":
                        if self.pos is not None:
                            self._close_at_open(i, "signal")
                    else:
                        if self.pos is not None:
                            if self.pos.side == o.side:
                                continue
                            self._close_at_open(i, "reverse")
                        if self._last_look_rejects(i, o.side):
                            self.pending.append(o)          # 拒否された注文は次の足で約定し直す
                            continue
                        self._open(i, o)
            if self.pos is not None and self.pos.entry_idx <= i:
                self._check_stops(i)
            self._mark(i)
            if self.equity <= 0:
                self.ruined = True
            self.equity_curve[i] = self.equity
            self.balance_curve[i] = self.balance
            if not self.ruined and i < n - 1:
                if self.force_close[i] and self.pos is not None:
                    self._submit(_Order("close"))
                elif i >= warmup and self.decide[i]:
                    self.strategy.on_bar(i, ctx)
            if progress is not None and (i % every == every - 1):
                progress(self, i)
        if self.pos is not None:
            i = n - 1
            px = self.bc[i] if self.pos.side > 0 else self.ac[i]
            self._close(i, px, self.c[i], "end")
            self.equity = self.balance
            self.equity_curve[i] = self.equity
            self.balance_curve[i] = self.balance
        if progress is not None:
            progress(self, n - 1)
        return self
