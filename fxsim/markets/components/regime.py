"""相場の局面(レジーム)。トレンド・レンジ・荒れ相場が予告なく入れ替わる。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..base import Component, GenContext, Stage, apply_block_increments, block_view

REGIME_NAMES = ("range", "trend_up", "trend_down", "turbulent")


@dataclass
class RegimeSpec:
    drift_annual: float     # 年率ドリフト(log)
    vol_mult: float
    mean_hours: float       # 平均継続時間


def _default_states():
    return {
        "range": {"drift_annual": 0.0, "vol_mult": 0.8, "mean_hours": 60.0},
        "trend_up": {"drift_annual": 0.20, "vol_mult": 1.0, "mean_hours": 110.0},
        "trend_down": {"drift_annual": -0.20, "vol_mult": 1.0, "mean_hours": 110.0},
        "turbulent": {"drift_annual": 0.0, "vol_mult": 1.7, "mean_hours": 24.0},
    }


def _default_transitions():
    return {
        "range": {"trend_up": 0.35, "trend_down": 0.35, "turbulent": 0.30},
        "trend_up": {"range": 0.55, "turbulent": 0.25, "trend_down": 0.20},
        "trend_down": {"range": 0.55, "turbulent": 0.25, "trend_up": 0.20},
        "turbulent": {"range": 0.50, "trend_up": 0.25, "trend_down": 0.25},
    }


class RegimeSwitching(Component):
    """局面ごとにドリフトとボラ倍率を変える。継続時間は指数分布(最短 min_hours)。

    meta["regime"] に局面番号(REGIME_NAMES の添字)の列を残す。後段の部品
    (レンジの平均回帰、ダマシの確率)がこれを読む。
    """

    key = "regime"
    stage = Stage.DRIFT

    @dataclass
    class Params:
        states: dict = field(default_factory=_default_states)
        transitions: dict = field(default_factory=_default_transitions)
        min_hours: float = 6.0

    def apply(self, ctx: GenContext) -> None:
        names = list(REGIME_NAMES)
        specs = {k: RegimeSpec(**v) for k, v in self.p.states.items()}
        steps_per_hour = 60 / ctx.base.step_minutes
        n = ctx.n
        regime = np.empty(n, dtype=np.int8)
        rng = ctx.rng
        state = names[int(rng.integers(len(names)))]
        pos = 0
        while pos < n:
            spec = specs[state]
            hours = max(self.p.min_hours, rng.exponential(spec.mean_hours))
            length = max(1, int(hours * steps_per_hour))
            regime[pos:pos + length] = names.index(state)
            ctx.event(pos, "regime", name=state)
            pos += length
            trans = self.p.transitions[state]
            keys = list(trans)
            probs = np.array([trans[k] for k in keys], dtype=float)
            state = keys[int(rng.choice(len(keys), p=probs / probs.sum()))]
        drift = np.array([specs[k].drift_annual for k in names]) / ctx.steps_per_year
        vm = np.array([specs[k].vol_mult for k in names])
        mult = vm[regime]
        mult /= np.sqrt(np.mean(mult * mult))
        ctx.drift += drift[regime]
        ctx.vol *= mult
        ctx.meta["regime"] = regime
        ctx.meta["regime_names"] = names


class RangeMeanReversion(Component):
    """レンジ局面では、局面開始時の価格に引き戻す力(OU)を掛ける。

    halflife_hours が短いほど強く戻る。ブロック(block_minutes)単位でフィードバックする。
    """

    key = "range_reversion"
    stage = Stage.POST_PATH

    @dataclass
    class Params:
        halflife_hours: float = 16.0
        block_minutes: int = 15

    def apply(self, ctx: GenContext) -> None:
        regime = ctx.meta.get("regime")
        if regime is None:
            return
        bs = max(1, self.p.block_minutes // ctx.base.step_minutes)
        starts, ends, _, _, close = block_view(ctx, bs)
        reg_b = regime[ends]
        kappa = 1 - 0.5 ** (self.p.block_minutes / 60 / self.p.halflife_hours)
        d = np.zeros(len(starts))
        off = 0.0
        anchor = None
        prev_reg = -1
        close_l = close.tolist()
        reg_l = reg_b.tolist()
        for b in range(len(close_l)):
            p = close_l[b] + off
            r = reg_l[b]
            if r != prev_reg:
                anchor = p
                prev_reg = r
            if r == 0:
                step = -kappa * (p - anchor)
                d[b] = step
                off += step
        apply_block_increments(ctx, d, bs)
