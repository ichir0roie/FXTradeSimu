"""MT5 の足でバックテストを回し、最後の確定足で戦略が出した注文を取り出す。

渡す足の最後の 1 本は、いま作られている足(まだ確定していない)。Backtester は足 i の終値で on_bar を呼び、
注文を足 i+1 の始値で約定させるので、最後の確定足(n-2)の注文は、最後の足(n-1)の始値で約定するはずのもの。
_SnapshotBacktester は足 n-1 の約定の直前に、その注文とシミュレーションの建玉を写し取る(約定はさせない)。

起点(最初の足)を固定して毎回同じ足から回すので、戦略の中の状態(前回の取引の結果など)は回すたびに同じになる。

本命の戦略(engine = "mt5"、strategies/main/)は MT5 のテスター(backtest/mt5)で回す。
テスターの成行はその場で約定するので、_SnapshotMt5Tester は、作りかけの足の始値のティックで
戦略を呼んだ前後の建玉を比べて、同じ SimOrder の形に直す(消えた建玉 → close / reduce、
増えた建玉 → enter。損切り・利確は約定値からの距離)。作りかけの足の中は進めない。
建玉は 1 つまで(両建てや複数の建玉を持つ戦略はまだ常駐できない)。
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field

from ..backtest.broker import BrokerConfig
from ..backtest.engine import Backtester, EngineConfig, Position
from ..backtest.mt5 import Mt5Tester
from ..backtest.schedule import TradingSchedule
from ..data import Bars


@dataclass
class SimOrder:
    """最後の確定足で戦略が出した注文。units は Backtester が約定のときに使う数量(通貨)。"""
    kind: str                    # enter | close | add | reduce
    side: int = 0
    units: float = 0.0
    sl_dist: float | None = None
    tp_dist: float | None = None
    sl_from_average: bool = False


@dataclass
class SimSnapshot:
    bar_t: int                          # 最後の確定足の開始時刻(UTC ns)
    position: Position | None           # 最後の確定足の終値の時点(on_bar で動かした損切り・利確を含む)
    position_t: int | None = None       # その建玉を建てた足の開始時刻(UTC ns)。同じ建玉かどうかの目印
    orders: list[SimOrder] = field(default_factory=list)
    equity: float = 0.0
    balance: float = 0.0
    halted: bool = False                # シミュレーションが破綻して取引を止めた
    n_trades: int = 0


class _SnapshotBacktester(Backtester):
    snapshot: SimSnapshot | None = None

    def _fill(self, i: int) -> None:
        last = len(self.bars) - 1
        if i < last:
            super()._fill(i)
            return
        orders = []
        for o in self.pending:
            units = 0.0
            if o.kind == "enter":
                price = self.ao[i] if o.side > 0 else self.bo[i]
                units = self._units(o.sl_dist, price, o.units)
            elif o.units is not None:
                units = float(o.units)
            orders.append(SimOrder(o.kind, o.side, units, o.sl_dist, o.tp_dist, o.sl_from_average))
        pos_t = int(self.t[self.pos.entry_idx]) if self.pos is not None else None
        self.snapshot = SimSnapshot(int(self.t[i - 1]), copy.copy(self.pos), pos_t, orders, self.equity,
                                    self.balance, self.halted, len(self.trades))
        self.pending = []           # いま作られている足では約定させない(実口座で約定させる)


class _Done(Exception):
    pass


class _SnapshotMt5Tester(Mt5Tester):
    snapshot: SimSnapshot | None = None
    _pre: dict | None = None

    def _process(self, k0: int, stop: int) -> None:
        last = len(self.bars) - 1
        if self.i < last:
            super()._process(k0, stop)
            return
        if k0 == 0:                 # 作りかけの足の始値(窓)で起きる発動までは進める
            super()._process(k0, stop)
            self._pre = {t: copy.copy(p) for t, p in self.positions.items()}
            return
        self.snapshot = self._take()  # 最後の確定足の判断(on_bar)の後。作りかけの足の中は進めない
        raise _Done

    def _take(self) -> SimSnapshot:
        pre, post = self._pre or {}, self.positions
        if len(pre) > 1 or len(post) > 1:
            raise ValueError("fxsim trade supports main strategies holding at most one position")
        cs = self.sym.contract_size
        orders: list[SimOrder] = []
        for t, p in pre.items():
            q = post.get(t)
            if q is None:
                orders.append(SimOrder("close", p.side))
            elif q.volume < p.volume - 1e-9:
                orders.append(SimOrder("reduce", p.side, (p.volume - q.volume) * cs))
        for t, q in post.items():
            if t not in pre:
                orders.append(SimOrder("enter", q.side, q.volume * cs,
                                       abs(q.price_open - q.sl) if q.sl is not None else None,
                                       abs(q.tp - q.price_open) if q.tp is not None else None))
        position, pos_t = None, None
        if pre:
            t, p = next(iter(pre.items()))
            p = post.get(t, p)          # 判断で動かした損切り・利確を含める(閉じたなら閉じる前)
            position = Position(p.side, p.volume * cs, p.price_open, p.open_mid, p.open_idx, p.sl, p.tp, p.swap,
                                p.commission, p.slippage)
            pos_t = int(self.t[p.open_idx])
        i = len(self.bars) - 1
        return SimSnapshot(int(self.t[i - 1]), position, pos_t, orders, self.balance + self.floating(),
                           self.balance, self.halted, len(self.trades))


def simulate(bars: Bars, strategy, broker: BrokerConfig, engine: EngineConfig, schedule: TradingSchedule,
             pip: float, seed: int = 0) -> SimSnapshot:
    """bars(最後の 1 本は作りかけの足)でバックテストを回し、最後の確定足の判断を返す。"""
    if len(bars) < 2:
        raise ValueError("need at least one closed bar and the forming bar")
    if getattr(strategy, "engine", "classic") == "mt5":
        mt = _SnapshotMt5Tester(bars, strategy, broker, engine, pip=pip, seed=seed, schedule=schedule)
        with contextlib.suppress(_Done):
            mt.run()
        if mt.snapshot is None:    # 途中で破綻して止まった
            return SimSnapshot(int(bars.t[-2]), None, None, [], mt.equity, mt.balance, True, len(mt.trades))
        return mt.snapshot
    bt = _SnapshotBacktester(bars, strategy, broker, engine, pip=pip, seed=seed, schedule=schedule)
    bt.run()
    if bt.snapshot is None:        # 途中で破綻して止まった(足 n-1 の約定まで来なかった)
        return SimSnapshot(int(bars.t[-2]), None, None, [], bt.equity, bt.balance, True, len(bt.trades))
    return bt.snapshot
