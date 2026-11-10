"""スプレッドの部品。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Component, GenContext, Stage

# UTC の時間帯ごとのスプレッド倍率。ロールオーバー(21-22 時)は大きく広がる。
DEFAULT_SPREAD_PROFILE = (
    1.3, 1.1, 1.1, 1.2, 1.3, 1.3,
    1.2, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.1, 1.2,
    1.3, 1.5, 2.0, 6.0, 4.0, 1.8,
)


class ConstantSpread(Component):
    key = "spread"
    stage = Stage.SPREAD

    @dataclass
    class Params:
        pips: float = 0.3

    def apply(self, ctx: GenContext) -> None:
        ctx.spread = np.full(ctx.n, self.p.pips * ctx.base.pip)


class DynamicSpread(Component):
    """時間帯・ボラ・指標・ストップ狩りでスプレッドが広がる。

    s = base * profile[hour] * (ボラ / ボラの中央値)^elasticity * 対数正規ノイズ
    指標発表の前後と、ストップ狩りの最中はさらに倍率を掛ける。週明けの最初の数十分も広い。
    ボラは時間帯の季節性を除いたもの(季節性はプロファイルで扱う)。
    """

    key = "spread"
    stage = Stage.SPREAD

    @dataclass
    class Params:
        base_pips: float = 0.3
        min_pips: float = 0.2
        max_pips: float = 40.0
        profile: tuple = DEFAULT_SPREAD_PROFILE
        vol_elasticity: float = 0.7
        noise_sigma: float = 0.15
        news_mult: float = 8.0
        news_before_min: int = 2
        news_after_min: int = 5
        hunt_mult: float = 2.5
        week_open_mult: float = 5.0
        week_open_min: int = 30

    def apply(self, ctx: GenContext) -> None:
        p = self.p
        sm = ctx.base.step_minutes
        prof = np.asarray(p.profile, dtype=float)[ctx.hours]
        v = ctx.vol / ctx.meta.get("session_mult", 1.0)
        rel = v / np.median(v)
        s = p.base_pips * prof * rel ** p.vol_elasticity
        s *= np.exp(ctx.rng.standard_normal(ctx.n) * p.noise_sigma - 0.5 * p.noise_sigma ** 2)
        mult = np.ones(ctx.n)
        for i in ctx.meta.get("news_idx", []):
            a = max(0, i - p.news_before_min // sm)
            mult[a:i + max(1, p.news_after_min // sm)] *= p.news_mult
        for i in ctx.meta.get("week_open_idx", []):
            mult[i:i + max(1, p.week_open_min // sm)] *= p.week_open_mult
        hunt = ctx.meta.get("hunt_mask")
        if hunt is not None:
            mult[hunt] *= p.hunt_mult
        s = np.clip(s * mult, p.min_pips, p.max_pips)
        ctx.spread = s * ctx.base.pip
