"""複数の通貨ペアを 1 つの口座で回すバックテスト(ペアごとにポジションは 1 つまで、ネット)。

ペアごとの約定・費用・スワップ・損切りは、1 ペアのエンジン(Backtester)をそのまま使う(Leg)。
残高と有効証拠金は全ペアで共有し、足の順番は 1 ペアのときと同じ:
    1. ペアごとに: スワップ、前の足の注文の約定、足の中の損切り・利確(Backtester._fill)
    2. ペアごとの評価損益を足して、口座の有効証拠金
    3. ロスカット: 有効証拠金が「全ペアの必要証拠金 x losscut_level」を割ったら、全ペアをその足の終値で決済する
       (1 ペアのエンジンは足の中の経路で判定するが、ここは口座全体なので足の終わりで見る。日足だけの相場では
       どちらも足の終わりになる)
    4. 破綻の判定(EngineConfig.ruin_frac)
    5. strategy.on_bar(i, ctx) を 1 回呼ぶ。ctx.target(ペア, 向き) で注文し、次の足の始値で約定する
建てられる量は「有効証拠金 x レバレッジ − ほかのペアの建玉の金額」まで。
全ペアが同じ決済通貨(円)である前提(損益をそのまま足す)。
"""
from __future__ import annotations

import numpy as np

from ..data import Bars, MarketData, load_rates
from ..data.rates import MULTI_RATES
from .broker import BrokerConfig
from .engine import Backtester, EngineConfig, ProgressFn, Trade, _Order
from .schedule import TradingSchedule


class _Account:
    def __init__(self, balance: float):
        self.balance = balance
        self.equity = balance


class _Idle:
    """Leg の中の戦略(使わない)。"""
    warmup = 0

    def prepare(self, bars, rng):
        pass


class Leg(Backtester):
    """1 ペア分。残高・有効証拠金は口座(_Account)と共有する。"""

    def __init__(self, pair: str, acct: _Account, book: list, *args, **kw):
        self.pair, self.acct, self.book = pair, acct, book
        self.unrealized = 0.0
        super().__init__(*args, **kw)

    @property
    def balance(self) -> float:
        return self.acct.balance

    @balance.setter
    def balance(self, v: float) -> None:
        self.acct.balance = v

    @property
    def equity(self) -> float:
        return self.acct.equity

    @equity.setter
    def equity(self, v: float) -> None:     # 有効証拠金は口座で計算する(ペア単位では上書きしない)
        pass

    def notional(self, i: int) -> float:
        return self.pos.units * self.c[i] if self.pos is not None else 0.0

    def _units(self, sl_dist, price, units):
        c = self.cfg
        if units is None:
            units = self.acct.equity * c.risk_fraction / sl_dist if (c.sizing == "risk" and sl_dist) else c.units
        ref = max(self.i - 1, 0)
        used = sum(leg.notional(ref) for leg in self.book if leg is not self)
        cap = max(0.0, self.acct.equity * self.broker.leverage - used) / price
        units = min(units, cap, c.max_units)
        return float(np.floor(units / 1000.0) * 1000.0)

    def _add(self, i, o):                       # 積み増しは使わない(複数ペアの戦略は向きだけ決める)
        raise NotImplementedError("add is not supported in portfolio backtests")

    def _losscut_price(self, i):                # ロスカットは口座全体で見る(PortfolioBacktester)
        return None

    def _close(self, i, price, mid, reason, slip=0.0):
        super()._close(i, price, mid, reason, slip)
        self.trades[-1].pair = self.pair

    def _mark(self, i: int) -> None:
        p = self.pos
        if p is None:
            self.unrealized = 0.0
            return
        px = self.bc[i] if p.side > 0 else self.ac[i]
        self.unrealized = (px - p.entry_price) * p.side * p.units

    def close_at_close(self, i: int, reason: str) -> None:
        p = self.pos
        if p is not None:
            self._close(i, self.bc[i] if p.side > 0 else self.ac[i], self.c[i], reason)


class PortfolioContext:
    """複数ペアの戦略から見えるもの。注文は次の足の始値で約定する。"""

    def __init__(self, pb: PortfolioBacktester):
        self._pb = pb

    @property
    def pairs(self) -> list[str]:
        return self._pb.pairs

    @property
    def bars(self) -> dict[str, Bars]:
        return self._pb.bars_by_pair

    @property
    def equity(self) -> float:
        return self._pb.equity

    @property
    def balance(self) -> float:
        return self._pb.balance

    @property
    def initial_balance(self) -> float:
        return float(self._pb.cfg.initial_balance)

    @property
    def base_units(self) -> float:
        return float(self._pb.cfg.units)

    @property
    def rng(self) -> np.random.Generator:
        return self._pb.strategy_rng

    @property
    def trades(self) -> list[Trade]:
        return self._pb.trades

    @property
    def can_enter(self) -> bool:
        return bool(self._pb.can_enter[self._pb.i])

    def side(self, pair: str) -> int:
        leg = self._pb.legs[pair]
        return leg.pos.side if leg.pos is not None else 0

    def position(self, pair: str):
        return self._pb.legs[pair].pos

    def target(self, pair: str, side: int, units: float | None = None) -> None:
        """pair を side(+1 買い / -1 売り / 0 持たない)にする。向きが同じなら何もしない(量は変えない)。
        取引時間外なら新規は出さない(逆向きの建玉の決済だけ行う)。"""
        leg = self._pb.legs[pair]
        cur = self.side(pair)
        side = int(np.sign(side))
        if side == cur:
            return
        leg.i = self._pb.i
        swap_ok = leg.swap_side_ok(side) if side else True
        if not swap_ok:
            leg.stats["swap_blocked"] += 1
        if side == 0:
            self._pb.submit(leg, _Order("close"))
        elif self.can_enter and swap_ok:
            self._pb.submit(leg, _Order("enter", side, None, None, units))
        elif cur == -side:
            self._pb.submit(leg, _Order("close"))

    def close_all(self) -> None:
        for pair in self.pairs:
            self.target(pair, 0)


class PortfolioBacktester:
    def __init__(self, bars_by_pair: dict[str, Bars], strategy, broker: BrokerConfig | None = None,
                 config: EngineConfig | None = None, pip: float = 0.01, seed: int = 0,
                 schedule: TradingSchedule | None = None, rates_path: str = MULTI_RATES):
        self.pairs = list(bars_by_pair)
        first = bars_by_pair[self.pairs[0]]
        for p, b in bars_by_pair.items():
            if len(b) != len(first) or not np.array_equal(b.t, first.t):
                raise ValueError(f"bars of {p} are not aligned with {self.pairs[0]}")
        self.bars_by_pair = bars_by_pair
        self.bars = first
        self.strategy = strategy
        self.broker = broker or BrokerConfig()
        self.cfg = config or EngineConfig()
        self.pip = pip
        self.acct = _Account(self.cfg.initial_balance)
        rates = load_rates(rates_path) if self.broker.swap_mode == "rates" else None
        book: list[Leg] = []
        self.legs: dict[str, Leg] = {}
        for j, pair in enumerate(self.pairs):
            b = bars_by_pair[pair]
            diff = rates.pair_differential(pair, b.t) if rates is not None else None
            leg = Leg(pair, self.acct, book, b, _Idle(), self.broker, self.cfg, pip=pip, seed=seed + 1_000_003 * j,
                      schedule=schedule, rate_diff=diff)
            book.append(leg)
            self.legs[pair] = leg
        lead = book[0]
        self.decide, self.can_enter, self.force_close = lead.decide, lead.can_enter, lead.force_close
        self.strategy_rng = np.random.default_rng([seed, 104729])
        n = len(first)
        self.equity_curve = np.full(n, np.nan)
        self.balance_curve = np.full(n, np.nan)
        self.trades: list[Trade] = []
        self._trade_ptr = {p: 0 for p in self.pairs}
        self.decisions: list[tuple] = []          # 先読み検査用 (i, pair, kind, side, units)
        self.i = 0
        self.ruined = False
        self.ruin_idx: int | None = None
        self.halted = False
        self.min_equity = self.acct.balance
        self.port_stats = {"losscut": 0}

    # ---- 口座 ----
    @property
    def balance(self) -> float:
        return self.acct.balance

    @property
    def equity(self) -> float:
        return self.acct.equity

    @property
    def stats(self) -> dict:
        out: dict = {}
        for leg in self.legs.values():
            for k, v in leg.stats.items():
                out[k] = out.get(k, 0) + v
        out["losscut"] = out.get("losscut", 0) + self.port_stats["losscut"]
        return out

    def submit(self, leg: Leg, order: _Order) -> None:
        leg.i = self.i
        units = None if order.units is None else round(float(order.units), 2)
        self.decisions.append((self.i, leg.pair, order.kind, order.side, units))
        leg._submit(order)

    def _collect_trades(self) -> None:
        for p, leg in self.legs.items():
            k = self._trade_ptr[p]
            if len(leg.trades) > k:
                self.trades.extend(leg.trades[k:])
                self._trade_ptr[p] = len(leg.trades)

    def _mark(self, i: int) -> None:
        for leg in self.legs.values():
            leg._mark(i)
        self.acct.equity = self.acct.balance + sum(leg.unrealized for leg in self.legs.values())
        margin = sum(leg.notional(i) for leg in self.legs.values()) / self.broker.leverage
        if margin > 0 and self.acct.equity < margin * self.broker.losscut_level:
            for leg in self.legs.values():
                if leg.pos is not None:
                    leg.close_at_close(i, "losscut")
                    self.port_stats["losscut"] += 1
            self._mark_flat()

    def _mark_flat(self) -> None:
        for leg in self.legs.values():
            leg.unrealized = 0.0
        self.acct.equity = self.acct.balance

    def _check_ruin(self, i: int) -> None:
        self.min_equity = min(self.min_equity, self.acct.equity)
        if not self.ruined and (self.acct.equity <= 0
                                or self.acct.equity <= self.cfg.initial_balance * self.cfg.ruin_frac):
            self.ruined = True
            self.ruin_idx = i
        if self.acct.equity <= 0:
            self.halted = True
        elif self.ruined and self.cfg.stop_at_ruin and not self.halted:
            for leg in self.legs.values():
                leg.close_at_close(i, "ruin")
                leg.pending = []
            self._mark_flat()
            self.halted = True

    def run(self, progress: ProgressFn | None = None, every: int = 500) -> PortfolioBacktester:
        n = len(self.bars)
        self.strategy.prepare(self.bars_by_pair, self.strategy_rng)
        ctx = PortfolioContext(self)
        warmup = int(getattr(self.strategy, "warmup", 0))
        legs = list(self.legs.values())
        for i in range(n):
            self.i = i
            for leg in legs:
                leg.i = i
            if self.halted:
                self.equity_curve[i] = self.acct.equity
                self.balance_curve[i] = self.acct.balance
                continue
            for leg in legs:
                leg._fill(i)
            self._mark(i)
            self._check_ruin(i)
            self._collect_trades()
            self.equity_curve[i] = self.acct.equity
            self.balance_curve[i] = self.acct.balance
            if not self.halted and i < n - 1:
                if self.force_close[i]:
                    ctx.close_all()
                elif i >= warmup and self.decide[i]:
                    self.strategy.on_bar(i, ctx)
            if progress is not None and (i % every == every - 1):
                progress(self, i)
        i = n - 1
        for leg in legs:
            leg.close_at_close(i, "end")
        self._mark_flat()
        self._collect_trades()
        self.equity_curve[i] = self.acct.equity
        self.balance_curve[i] = self.acct.balance
        if progress is not None:
            progress(self, n - 1)
        return self



def build_backtester(md: MarketData, strategy, broker: BrokerConfig, config: EngineConfig, *, seed: int,
                     schedule: TradingSchedule | None = None) -> Backtester | PortfolioBacktester:
    """相場と戦略から、バックテストを組み立てる。複数ペアの戦略は相場の legs(basket_*)の全ペアで回す。
    MT5 前提の戦略(engine = "mt5"、strategies/main/)は MT5 のテスター(backtest/mt5)で回す。"""
    tf = strategy.timeframe
    # 相場がペアを名乗るとき(real_pair)は、スワップにそのペアの金利差を使う(省くと USD − JPY の政策金利)
    pair = md.meta.get("swap_pair")
    diff = None
    if pair and not strategy.multi and broker.swap_mode == "rates":
        diff = load_rates(MULTI_RATES).pair_differential(pair, md.bars(tf).t)
    if getattr(strategy, "engine", "classic") == "mt5":
        from .mt5 import Mt5Tester

        if strategy.multi:
            raise ValueError("multi-pair MT5 strategies are not supported yet")
        return Mt5Tester(md.bars(tf), strategy, broker, config, pip=md.pip, seed=seed, schedule=schedule,
                         rate_diff=diff, symbol=pair.upper() if pair else "USDJPY")
    if strategy.multi:
        legs = md.meta.get("legs")
        if not legs:
            raise ValueError("multi-pair strategy needs a basket market (basket_*)")
        return PortfolioBacktester({p: leg.bars(tf) for p, leg in legs.items()}, strategy, broker, config,
                                   pip=md.pip, seed=seed, schedule=schedule)
    return Backtester(md.bars(tf), strategy, broker, config, pip=md.pip, seed=seed, schedule=schedule,
                      rate_diff=diff)
