"""バックテストのエンジン(ポジションは常に 1 つまで、ネット)。

1 本の足 i で起きる順番:
    1. ロールオーバーをまたいだ分のスワップ
    2. 前の足の終わりに出した注文を、足 i の始値で約定(買いは ask、売りは bid)
    3. 足 i の中で損切り/利確を判定(両方届いたら損切りを先にする = 悲観的)
    4. 終値で評価損益、ロスカット判定
    5. strategy.on_bar(i) を呼ぶ(ここで出した注文は次の足の始値で約定する)
戦略は足 i の終値までしか見られないので、先読みは構造的に起きない
(ただし prepare で未来を使う指標を作らないこと。backtest.lookahead で検査できる)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..calendar import rollover_weight, trading_day
from ..data import Bars
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


@dataclass
class Trade:
    side: int
    units: float
    entry_t: int
    exit_t: int
    entry_price: float
    exit_price: float
    pnl: float            # 円(スワップ込み)
    pips: float           # 約定価格どうしの pip(コスト込み)
    gross_pips: float     # mid どうしの pip(コスト抜き = 戦略の素の優位性)
    cost_pips: float      # gross - pips(スプレッド + スリッページ)
    swap: float
    reason: str           # signal | sl | tp | hunted | losscut | end | reverse
    bars_held: int


@dataclass
class _Order:
    kind: str                # enter | close
    side: int = 0
    sl_dist: float | None = None
    tp_dist: float | None = None
    units: float | None = None


class StrategyContext:
    """戦略から見えるもの。注文は次の足の始値で約定する。"""

    def __init__(self, engine: "Backtester"):
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


class Backtester:
    def __init__(self, bars: Bars, strategy, broker: BrokerConfig | None = None,
                 config: EngineConfig | None = None, pip: float = 0.01, seed: int = 0,
                 schedule: TradingSchedule | None = None):
        self.bars = bars
        self.schedule = schedule or TradingSchedule()
        self.decide, self.can_enter, self.force_close = self.schedule.masks(bars.t, bars.timeframe)
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
        self.stats = {"hunted": 0, "losscut": 0, "swap_total": 0.0, "slippage_pips": 0.0}

    # ---- 注文の受付 ----
    def _submit(self, order: _Order) -> None:
        self.decisions.append((self.i, order.kind, order.side))
        if order.kind == "enter":
            self.pending = [o for o in self.pending if o.kind != "enter"]
        self.pending.append(order)

    # ---- 約定まわり ----
    def _slip(self, i: int, stop: bool = False) -> float:
        b = self.broker
        size = b.slippage_pips + b.slippage_spread_k * self.bars.spread_mean[i] / self.pip
        if size <= 0:
            return 0.0
        z = self.rng.standard_normal()
        if b.adverse_only:
            z = abs(z)
        s = z * size * (b.stop_slippage_mult if stop else 1.0)
        self.stats["slippage_pips"] += s
        return s * self.pip   # 正 = 不利

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

    def _open(self, i: int, o: _Order) -> None:
        bars = self.bars
        mid = bars.open[i]
        if o.side > 0:
            price = bars.ask_open[i] + self._slip(i)
        else:
            price = bars.bid_open[i] - self._slip(i)
        units = self._units(o.sl_dist, price, o.units)
        if units < self.cfg.min_units:
            return
        sl = price - o.side * o.sl_dist if o.sl_dist else None
        tp = price + o.side * o.tp_dist if o.tp_dist else None
        self.pos = Position(o.side, units, price, mid, i, sl, tp)

    def _close(self, i: int, price: float, mid: float, reason: str) -> None:
        p = self.pos
        pips = (price - p.entry_price) * p.side / self.pip
        gross = (mid - p.entry_mid) * p.side / self.pip
        pnl = (price - p.entry_price) * p.side * p.units + p.swap
        self.balance += pnl - p.swap   # スワップは付いた時点で残高に入れている
        self.trades.append(Trade(p.side, p.units, int(self.bars.t[p.entry_idx]), int(self.bars.t[i]),
                                 p.entry_price, price, pnl, pips, gross, gross - pips, p.swap, reason,
                                 i - p.entry_idx))
        self.pos = None

    def _close_at_open(self, i: int, reason: str) -> None:
        bars = self.bars
        if self.pos.side > 0:
            price = bars.bid_open[i] - self._slip(i)
        else:
            price = bars.ask_open[i] + self._slip(i)
        self._close(i, price, bars.open[i], reason)

    def _check_stops(self, i: int) -> None:
        p = self.pos
        bars = self.bars
        half = bars.spread_mean[i] / 2
        if p.side > 0:
            o_, lo, hi = bars.bid_open[i], bars.bid_low[i], bars.bid_high[i]
        else:   # 売りは ask で評価し、符号を反転して買いと同じ式にする
            o_, lo, hi = -bars.ask_open[i], -bars.ask_high[i], -bars.ask_low[i]
        sl = p.sl * p.side if p.sl is not None else None
        tp = p.tp * p.side if p.tp is not None else None
        # 窓(始値で既に越えている)
        if sl is not None and o_ <= sl:
            fill = o_ - self._slip(i, stop=True)
            self._close(i, fill * p.side, fill * p.side + p.side * half, "sl")
            return
        if tp is not None and o_ >= tp:
            fill = o_ if self.broker.positive_slippage else tp
            self._close(i, fill * p.side, fill * p.side + p.side * half, "tp")
            return
        hit_sl = sl is not None and lo <= sl
        reason = "sl"
        if (sl is not None and not hit_sl and self.broker.hunt_prob > 0
                and lo - sl <= self.broker.hunt_pips * self.pip and self.rng.random() < self.broker.hunt_prob):
            hit_sl = True
            reason = "hunted"
            self.stats["hunted"] += 1
        if hit_sl:   # 両方届いた足は損切りが先(悲観的)
            fill = sl - self._slip(i, stop=True)
            self._close(i, fill * p.side, (sl + half) * p.side, reason)
            return
        if tp is not None and hi >= tp:
            self._close(i, tp * p.side, (tp + half) * p.side, "tp")

    def _swap(self, i: int) -> None:
        if i == 0 or self.pos is None:
            return
        d0 = int(trading_day(self.bars.t[i - 1]))
        d1 = int(trading_day(self.bars.t[i]))
        if d1 <= d0:
            return
        days = sum(rollover_weight(d) for d in range(d0, d1))
        rate = self.broker.swap_long_per_10k if self.pos.side > 0 else self.broker.swap_short_per_10k
        amt = rate * days * self.pos.units / 10_000
        self.pos.swap += amt
        self.balance += amt
        self.stats["swap_total"] += amt

    def _mark(self, i: int) -> None:
        p = self.pos
        if p is None:
            self.equity = self.balance
            return
        px = self.bars.bid_close[i] if p.side > 0 else self.bars.ask_close[i]
        self.equity = self.balance + (px - p.entry_price) * p.side * p.units
        margin = p.units * self.bars.close[i] / self.broker.leverage
        if self.equity < margin * self.broker.losscut_level:
            self._close(i, px, self.bars.close[i], "losscut")
            self.stats["losscut"] += 1
            self.equity = self.balance

    # ---- 実行 ----
    def run(self, progress: ProgressFn | None = None, every: int = 500) -> "Backtester":
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
            px = self.bars.bid_close[i] if self.pos.side > 0 else self.bars.ask_close[i]
            self._close(i, px, self.bars.close[i], "end")
            self.equity = self.balance
            self.equity_curve[i] = self.equity
            self.balance_curve[i] = self.balance
        if progress is not None:
            progress(self, n - 1)
        return self
