"""本命の戦略の土台(MT5 の EA として回す)。

サブクラスは signals(bars, rng) で、足ごとの 3 つの列を因果的に計算するだけでよい(足 i の終値までで決まる値):
    target[i]          持ちたい向き(+1 買い / −1 売り / 0 持たない)
    stop_dist[i]       損切りまでの幅(価格)。量はこの幅で決まる(guards.size_lots)
    expected_move[i]   1 回の取引で見込む値幅(価格)。コストゲート(guards.cost_ok)に使う
売買の決まり(on_bar、新しい足の最初のティック):
    0. 口座の床: 月初の有効証拠金 x protect を割っていたら、この戦略の建玉を全部手仕舞い、その月は新規を出さない
       (guards.EquityFloor。テスターは足の確定でだけ見る。常駐は fxsim trade が口座全体で見る)
    1. 終値の損切り: 確定した足の終値(買いは bid、売りは ask。stop_price = mid なら mid)が損切りの水準を越えていたら
       決済する(close_stop)。trail なら、水準を有利な向きにだけ動かす
    スプレッドのゲート(max_spread_mult > 0): 今のスプレッドが典型の何倍かを超えていたら、2 の信号の手仕舞いと 3 の新規を
       出さない(次の足でやり直す)。損切り・週末・床の手仕舞いは出す
    2. 手仕舞い: 向きが target と違い、min_hold_bars を過ぎていたら決済する。金曜の最後の足で手仕舞う設定もある
    3. 新規: 建玉が無く target ≠ 0 なら、取引時間・ロールオーバーの前後・ドローダウンの止め・回転の上限・
       コストゲートを通ったときだけ、
       guards.size_lots の量を、口座の全建玉が逆指値まで飛んでも床を割らない量(guards.floor_lots)に抑えて成行。
       close_stop なら非常用の逆指値(損切りの幅 x emergency_stop_mult)だけをサーバーに置く
方向でたらめ版(検証の帰無、main_direction_null): null_u を渡すと、新規の向きだけを乱数で決め、
手仕舞いの時期は元の向きの信号で決める(売買の時期・量・損切りの幅・コストゲートは同じ)。
数量・価格・注文は MT5 の決まり(ロット、ポイント、stops_level)で出す。
magic は戦略の名前から決まる(同じ口座のほかの EA と分ける)。
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..base import Strategy
from ..guards import (
    DrawdownBreaker,
    EquityFloor,
    GuardConfig,
    TurnoverBudget,
    cost_ok,
    floor_lots,
    last_decision_of_week,
    near_rollover,
    open_worst_loss,
    size_lots,
    spread_ok,
    typical_spread,
)


class MainStrategy(Strategy):
    tier: ClassVar[str] = "main"
    engine: ClassVar[str] = "mt5"

    @dataclass
    class Params:
        timeframe: int = 1440
        guard: dict = field(default_factory=dict)       # GuardConfig の上書き

    def __init__(self, **params):
        super().__init__(**params)
        self.g = GuardConfig.from_dict(getattr(self.p, "guard", {}))
        self.magic = zlib.crc32(self.name.encode()) % 1_000_000_000
        self.null_u: np.ndarray | None = None       # 方向でたらめ版のときの乱数(main_direction_null が入れる)

    # ---- サブクラスが書く ----
    def signals(self, bars, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    # ---- 共通 ----
    def prepare(self, bars, rng):
        target, stop_dist, move = self.signals(bars, rng)
        n = len(bars)
        self.target = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0).astype(int)
        self.stop_dist = np.asarray(stop_dist, dtype=float)
        self.expected_move = np.asarray(move, dtype=float)
        if not (len(self.target) == len(self.stop_dist) == len(self.expected_move) == n):
            raise ValueError("signals must return arrays as long as bars")
        self.bars = bars
        self.breaker = DrawdownBreaker(self.g)
        self.budget = TurnoverBudget(self.g, self.magic)
        self.floor = EquityFloor(self.g.protect)
        self.floor_cut = 0                         # 床のために量を減らした新規の回数
        self.stops: dict[int, float] = {}          # チケット -> 終値で判定する損切りの水準
        # チケット -> 建てたときの信号の向き(方向でたらめ版では実際の向きと違う)
        self.signal_side: dict[int, int] = {}
        self.blocked: dict[str, int] = {"session": 0, "drawdown": 0, "turnover": 0, "cost": 0, "size": 0,
                                    "floor": 0, "spread": 0, "spread_exit": 0}
        self.typical = (typical_spread(bars.spread_mean, self.g.spread_window) if self.g.max_spread_mult > 0
                        else np.full(n, np.nan))

    def on_bar(self, i, ctx):
        g, b = self.g, self.bars
        # 0. 口座の床(月初の有効証拠金 x protect)
        if g.protect > 0:
            self.floor.update(ctx.time, ctx.equity)
            if self.floor.check(ctx.equity):
                for p in ctx.positions(self.magic):
                    ctx.position_close(p.ticket, reason="floor")
                    self.stops.pop(p.ticket, None)
                return
        positions = ctx.positions(self.magic)
        # 1. 終値の損切り(と追いかけ)
        for p in positions:
            lvl = self.stops.get(p.ticket)
            if lvl is None:
                continue
            if g.stop_price == "mid":
                close = b.close[i]
            else:
                close = b.bid_close[i] if p.side > 0 else b.ask_close[i]
            if (close - lvl) * p.side <= 0:
                ctx.position_close(p.ticket, reason="close_stop")
                self.stops.pop(p.ticket, None)
            elif g.trail and np.isfinite(self.stop_dist[i]):
                self.stops[p.ticket] = max(lvl * p.side, (close - p.side * self.stop_dist[i]) * p.side) * p.side
        positions = ctx.positions(self.magic)
        want = int(self.target[i])
        weekend = g.flat_before_weekend and last_decision_of_week(b.t, i)
        wide = not spread_ok(ctx.ask - ctx.bid, float(self.typical[i]), g)
        # 2. 手仕舞い(信号の向きで判定する)。スプレッドが開いていれば信号の手仕舞いは次の足に回す
        for p in positions:
            sig = self.signal_side.get(p.ticket, p.side)
            if sig != want and p.bars_held >= g.min_hold_bars and wide and not weekend:
                self.blocked["spread_exit"] += 1
                continue
            if (sig != want and p.bars_held >= g.min_hold_bars) or weekend:
                ctx.position_close(p.ticket, reason="weekend" if weekend else "signal")
                self.stops.pop(p.ticket, None)
        if ctx.positions(self.magic) or want == 0 or weekend or not ctx.can_enter:
            return
        # 3. 新規
        signal = want
        if self.null_u is not None:
            want = signal if self.null_u[i] < 0.5 else -signal
        if near_rollover(ctx.time, g.avoid_rollover_hours) and b.timeframe < 1440:
            self.blocked["session"] += 1
            return
        if wide:
            self.blocked["spread"] += 1
            return
        if not self.breaker.allows(ctx, i):
            self.blocked["drawdown"] += 1
            return
        if not self.budget.allows(ctx):
            self.blocked["turnover"] += 1
            return
        if not cost_ok(ctx, want, self.expected_move[i], g):
            self.blocked["cost"] += 1
            return
        dist = float(self.stop_dist[i])
        lots = size_lots(ctx, dist, g, want)
        if lots <= 0:
            self.blocked["size"] += 1
            return
        entry = ctx.ask if want > 0 else ctx.bid
        level = entry - want * dist
        sl = None
        if not g.close_stop:
            sl = level
        elif g.emergency_stop_mult > 0:
            sl = entry - want * dist * g.emergency_stop_mult
        if g.protect > 0:
            # 口座の全建玉と、この新規が逆指値まで飛んで窓で滑っても、床を割らない量まで
            sym = ctx.symbol()
            room = self.floor.room(ctx.equity, open_worst_loss(ctx, g))
            per_lot = (abs(entry - sl) * sym.contract_size * g.gap_mult) if sl is not None else float("inf")
            capped = floor_lots(lots, room, per_lot, sym.volume_step)
            self.floor_cut += capped < lots
            lots = capped
            if lots < sym.volume_min:
                self.blocked["floor"] += 1
                return
        r = ctx.buy(lots, sl=sl) if want > 0 else ctx.sell(lots, sl=sl)
        if r.ok:
            for p in ctx.positions(self.magic):
                self.signal_side.setdefault(p.ticket, signal)
                if g.close_stop:
                    self.stops.setdefault(p.ticket, level)


__all__ = ["MainStrategy"]
