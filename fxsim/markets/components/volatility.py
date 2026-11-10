"""ボラティリティの部品: 時間帯の季節性と、ボラの塊(クラスタリング)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...numerics import ar1_filter
from ..base import Component, GenContext, Stage

# UTC の時間帯ごとの相対ボラ。東京は静か、ロンドン勢の参入と NY との重なりで荒れ、
# ロールオーバー前後(21-22 時)は板が薄くて静か。
DEFAULT_SESSION_PROFILE = (
    0.75, 0.80, 0.75, 0.65, 0.60, 0.60,   # 0-5   東京
    0.75, 1.10, 1.30, 1.20, 1.10, 1.05,   # 6-11  ロンドン
    1.45, 1.60, 1.50, 1.35, 1.05, 0.90,   # 12-17 NY と重なる
    0.80, 0.70, 0.60, 0.40, 0.45, 0.65,   # 18-23 NY 午後〜ロールオーバー
)


class SessionSeasonality(Component):
    """時間帯ごとにボラの倍率を掛ける。全体の分散は変えない(二乗平均で正規化)。"""

    key = "session"
    stage = Stage.VOLATILITY

    @dataclass
    class Params:
        profile: tuple = DEFAULT_SESSION_PROFILE

    def apply(self, ctx: GenContext) -> None:
        prof = np.asarray(self.p.profile, dtype=float)
        if prof.shape != (24,):
            raise ValueError("session.profile must have 24 values")
        m = prof[ctx.hours]
        m = m / np.sqrt(np.mean(m * m))
        ctx.vol *= m
        ctx.meta["session_mult"] = m


class StochasticVolatility(Component):
    """複数の時間スケールを持つ log ボラの AR(1) の和。荒れた相場はしばらく荒れ続ける。

    mult = exp(x - var(x)) とすることで E[mult^2] = 1 になり、平均の分散は変わらない。
    halflives は分、stds は各因子の log ボラの定常標準偏差。
    """

    key = "stoch_vol"
    stage = Stage.VOLATILITY

    @dataclass
    class Params:
        halflives_min: tuple = (45.0, 1440.0, 1440.0 * 15)
        stds: tuple = (0.25, 0.25, 0.30)

    def apply(self, ctx: GenContext) -> None:
        x = np.zeros(ctx.n)
        var = 0.0
        for hl_min, sd in zip(self.p.halflives_min, self.p.stds, strict=True):
            hl = max(hl_min / ctx.base.step_minutes, 1.0)
            a = 0.5 ** (1.0 / hl)
            e = ctx.rng.standard_normal(ctx.n) * sd * np.sqrt(1 - a * a)
            x += ar1_filter(e, a, x0=ctx.rng.standard_normal() * sd)
            var += sd * sd
        mult = np.exp(x - var)
        ctx.vol *= mult
        ctx.meta["sv_mult"] = mult
