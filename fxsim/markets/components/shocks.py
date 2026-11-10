"""ショックの部品: 裾の厚い分布、指標発表、週明けの窓。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...calendar import day_of_month, minute_of_day, weekday
from ...numerics import student_t_unit
from ..base import Component, GenContext, Stage


class FatTails(Component):
    """ショックを分散 1 の t 分布にする(df が小さいほど裾が厚い)。"""

    key = "fat_tails"
    stage = Stage.INNOVATION

    @dataclass
    class Params:
        df: float = 4.0

    def apply(self, ctx: GenContext) -> None:
        ctx.shocks = student_t_unit(ctx.rng, self.p.df, ctx.n)


class NewsShocks(Component):
    """経済指標。雇用統計(第 1 金曜 12:30 UTC)と、ランダムな指標が週に数回。

    - 発表時に t 分布のジャンプ(pip)
    - 発表後しばらくボラが跳ね上がり、指数的に減衰
    - whipsaw_prob の確率で「一度逆に振ってから本命の方向へ」(両方向の逆指値を狩る)
    - meta["news_idx"] にステップ位置を残す(スプレッド部品が発表前後を広げる)
    """

    key = "news"
    stage = Stage.VOLATILITY

    @dataclass
    class Params:
        nfp: bool = True
        nfp_pips: float = 35.0
        random_per_week: float = 2.5
        random_pips: float = 12.0
        candidate_minutes: tuple = (8 * 60, 9 * 60 + 30, 12 * 60 + 30, 14 * 60, 23 * 60 + 50)
        df: float = 3.0
        vol_boost: float = 4.0
        vol_decay_min: float = 20.0
        whipsaw_prob: float = 0.4
        whipsaw_frac: float = 0.6

    def apply(self, ctx: GenContext) -> None:
        t = ctx.times
        rng = ctx.rng
        mod = minute_of_day(t)
        dow = weekday(t)
        idx_list = []
        scales = []
        if self.p.nfp:
            nfp = np.flatnonzero((dow == 4) & (day_of_month(t) <= 7) & (mod == 12 * 60 + 30))
            idx_list.extend(nfp.tolist())
            scales.extend([self.p.nfp_pips] * len(nfp))
        cand = np.flatnonzero(np.isin(mod, self.p.candidate_minutes) & (dow <= 4))
        weeks = ctx.n * ctx.base.step_minutes / (5 * 1440)
        k = min(len(cand), rng.poisson(self.p.random_per_week * weeks))
        if k:
            pick = rng.choice(cand, size=k, replace=False)
            idx_list.extend(pick.tolist())
            scales.extend([self.p.random_pips] * k)
        if not idx_list:
            ctx.meta["news_idx"] = np.zeros(0, dtype=np.int64)
            return
        order = np.argsort(idx_list)
        idx = np.asarray(idx_list)[order]
        sc = np.asarray(scales)[order]
        decay_steps = max(1.0, self.p.vol_decay_min / ctx.base.step_minutes)
        L = int(decay_steps * 6)
        kern = 1 + self.p.vol_boost * np.exp(-np.arange(L) / decay_steps)
        for i, s in zip(idx.tolist(), sc.tolist()):
            m = min(L, ctx.n - i)
            ctx.vol[i:i + m] *= kern[:m]
            jump = s * student_t_unit(rng, self.p.df, 1)[0]
            j = ctx.pips_to_log(jump)
            if rng.random() < self.p.whipsaw_prob and i + 4 < ctx.n:
                ctx.jumps[i] += -self.p.whipsaw_frac * j
                ctx.jumps[i + 2:i + 5] += (1 + self.p.whipsaw_frac) * j / 3
                ctx.event(i, "news", pips=float(jump), whipsaw=True)
            else:
                ctx.jumps[i] += j
                ctx.event(i, "news", pips=float(jump), whipsaw=False)
        ctx.meta["news_idx"] = idx.astype(np.int64)


class WeekendGap(Component):
    """週明けの窓開け。金曜終値から日曜オープンまでの間に t 分布のジャンプ。"""

    key = "weekend_gap"
    stage = Stage.RETURNS

    @dataclass
    class Params:
        gap_pips: float = 10.0
        df: float = 3.0

    def apply(self, ctx: GenContext) -> None:
        gaps = np.flatnonzero(np.diff(ctx.times) > 3_600_000_000_000) + 1
        if len(gaps) == 0:
            return
        g = self.p.gap_pips * student_t_unit(ctx.rng, self.p.df, len(gaps))
        ctx.jumps[gaps] += ctx.pips_to_log(g)
        for i, x in zip(gaps.tolist(), g.tolist()):
            if abs(x) >= 2 * self.p.gap_pips:
                ctx.event(i, "weekend_gap", pips=float(x))
        ctx.meta["week_open_idx"] = gaps
