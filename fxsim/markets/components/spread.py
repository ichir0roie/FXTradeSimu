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


class RecordedSpread(Component):
    """実データ(MT5 の足)に記録されたスプレッドを使う(real_pair)。

    MT5 の足のスプレッドは足の中の最小なので、広がった瞬間を取りこぼす。とくにロールオーバー(夏 21 時・冬 22 時 UTC)の
    直前の足は狭いまま記録され、その足の終値で約定すると広がったスプレッドを払わずに済んでしまう(E015)。
    そこで足の境目の点(始値の点は前の足、終値の点は次の足)に、隣の足の記録との最大を使う。足の中の点(高値・安値)は
    その足の記録のまま(広がるのは境目の前後の数分なので、足全体を広げると損切りが刺さりすぎる)。
    記録が無い・下限より狭い点は floor_pips。1 本 1 点の相場(_bar_points が無い)では前後の足の最大。
    hour_profile なら、同じ銘柄の 1 分足の UTC の時ごとの平均(_spread_profile)も下限にする(v2。1 時間足・日足の記録は
    足の中の最小なので、ふだんの幅より狭い。日足の始値の点はロールオーバーの時間の平均になる)。
    """

    key = "spread"
    stage = Stage.SPREAD

    @dataclass
    class Params:
        floor_pips: float = 0.1      # 記録が無い・これより狭い点の下限(pip)
        hour_profile: bool = True    # 1 分足の時ごとの平均を下限にする(False で real_pair v1 と同じ)

    def apply(self, ctx: GenContext) -> None:
        rec = ctx.meta.get("_recorded_spread")
        floor = self.p.floor_pips * ctx.base.pip
        if rec is None:
            ctx.spread = np.full(ctx.n, floor)
            return
        k = int(ctx.meta.get("_bar_points", 1))
        per_bar = np.nan_to_num(np.asarray(rec, dtype=float)[::k], nan=0.0)
        prev = np.r_[per_bar[:1], per_bar[:-1]]
        nxt = np.r_[per_bar[1:], per_bar[-1:]]
        pts = np.repeat(per_bar[:, None], k, axis=1)
        if k == 1:
            pts[:, 0] = np.maximum(per_bar, np.maximum(prev, nxt))
        else:
            pts[:, 0] = np.maximum(per_bar, prev)          # 始値の点
            pts[:, -1] = np.maximum(per_bar, nxt)          # 終値の点
        out = np.maximum(pts.ravel(), floor)[:ctx.n]
        prof = ctx.meta.get("_spread_profile")
        if self.p.hour_profile and prof is not None:
            out = np.maximum(out, np.asarray(prof, dtype=float)[:ctx.n])
        ctx.spread = out


class DynamicSpread(Component):
    """時間帯・ボラ・指標・ストップ狩りでスプレッドが広がる。

    s = base * profile[hour] * (ボラ / ボラの中央値)^elasticity * 対数正規ノイズ
    指標発表の前後と、ストップ狩りの最中はさらに倍率を掛ける。週明けの最初の数十分も広い。
    薄商いの期間(HolidayLiquidity の thin_mask)は thin_mult 倍、異常な値動きの最中は
    ExtremeEvents が残す extreme_spread_mult の倍率を掛ける。
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
        thin_mult: float = 2.5

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
        thin = ctx.meta.get("thin_mask")
        if thin is not None:
            mult[thin] *= p.thin_mult
        extreme = ctx.meta.get("extreme_spread_mult")
        if extreme is not None:
            mult *= extreme
        s = np.clip(s * mult, p.min_pips, p.max_pips)
        ctx.spread = s * ctx.base.pip
