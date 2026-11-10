"""本命の戦略の土台(MT5 の EA として回す)。

サブクラスは signals(bars, rng) で、足ごとの 3 つの列を因果的に計算するだけでよい(足 i の終値までで決まる値):
    target[i]          持ちたい向き(+1 買い / −1 売り / 0 持たない)
    stop_dist[i]       損切りまでの幅(価格)。量はこの幅で決まる(guards.size_lots)
    expected_move[i]   1 回の取引で見込む値幅(価格)。費用の門(guards.cost_ok)に使う
売買の決まり(on_bar、新しい足の最初のティック):
    1. 終値の損切り: 確定した足の終値(買いは bid、売りは ask)が損切りの水準を越えていたら決済する(close_stop)。
       trail なら、水準を有利な向きにだけ動かす
    2. 手仕舞い: 向きが target と違い、min_hold_bars を過ぎていたら決済する。金曜の最後の足で手仕舞う設定もある
    3. 新規: 建玉が無く target ≠ 0 なら、取引時間・ロールオーバーの前後・ドローダウンの止め・回転の上限・
       費用の門を通ったときだけ、
       guards.size_lots の量で成行。close_stop なら非常用の逆指値(損切りの幅 x emergency_stop_mult)だけをサーバーに置く
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
    GuardConfig,
    TurnoverBudget,
    cost_ok,
    last_decision_of_week,
    near_rollover,
    size_lots,
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
        self.g = GuardConfig.from_dict(self.p.guard)
        self.magic = zlib.crc32(self.name.encode()) % 1_000_000_000

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
        self.stops: dict[int, float] = {}          # チケット -> 終値で判定する損切りの水準
        self.blocked: dict[str, int] = {"session": 0, "drawdown": 0, "turnover": 0, "cost": 0, "size": 0}

    def on_bar(self, i, ctx):
        g, b = self.g, self.bars
        positions = ctx.positions(self.magic)
        # 1. 終値の損切り(と追いかけ)
        for p in positions:
            lvl = self.stops.get(p.ticket)
            if lvl is None:
                continue
            close = b.bid_close[i] if p.side > 0 else b.ask_close[i]
            if (close - lvl) * p.side <= 0:
                ctx.position_close(p.ticket, reason="close_stop")
                self.stops.pop(p.ticket, None)
            elif g.trail and np.isfinite(self.stop_dist[i]):
                self.stops[p.ticket] = max(lvl * p.side, (close - p.side * self.stop_dist[i]) * p.side) * p.side
        positions = ctx.positions(self.magic)
        want = int(self.target[i])
        weekend = g.flat_before_weekend and last_decision_of_week(b.t, i)
        # 2. 手仕舞い
        for p in positions:
            if (p.side != want and p.bars_held >= g.min_hold_bars) or weekend:
                ctx.position_close(p.ticket, reason="weekend" if weekend else "signal")
                self.stops.pop(p.ticket, None)
        if ctx.positions(self.magic) or want == 0 or weekend or not ctx.can_enter:
            return
        # 3. 新規
        if near_rollover(ctx.time, g.avoid_rollover_hours) and b.timeframe < 1440:
            self.blocked["session"] += 1
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
        r = ctx.buy(lots, sl=sl) if want > 0 else ctx.sell(lots, sl=sl)
        if r.ok and g.close_stop:
            for p in ctx.positions(self.magic):
                self.stops.setdefault(p.ticket, level)


__all__ = ["MainStrategy"]
