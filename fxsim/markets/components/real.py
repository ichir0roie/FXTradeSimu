"""実際の相場を元にする部品。

RealReplay   実データをそのまま再生する(シードで変わらない)
RealSkeleton 実データの足(既定は日足)を「骨格」にし、1 分足の時刻列を用意する。
             mode="replay"    骨格 = 実データそのまま
             mode="bootstrap" 骨格 = 実データのリターンをブロック単位で並べ替えたもの(別の相場)
AnchorBridge 合成した経路を骨格に合わせる。骨格の終値を必ず通り、OHLC があれば
             各足の高値・安値にちょうど届くよう、はみ出しを伸縮する。
             骨格の足の中の動き(季節性・ボラの塊・指標・悪意)はシードごとに変わる。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...calendar import NS_DAY, NS_MIN, trading_times
from ..base import BaseParams, Component, GenContext, Stage
from ..real import RealSeries, load_ohlc

DEFAULT_DATASET = "data/real/usdjpy_d1_fred.csv"


@dataclass
class _SourceParams:
    path: str = DEFAULT_DATASET
    format: str = "auto"
    start: str | None = None      # 期間。未指定なら実験の years だけ末尾から取る
    end: str | None = None


def _series(p, years: float | None, days: float | None) -> RealSeries:
    s = load_ohlc(p.path, p.format)
    if days is not None and years is None:
        years = days / 365.25
    w = s.window(p.start, p.end, None if p.start else years)
    if len(w) < 3:
        raise ValueError(f"real data window too short: {len(w)} rows ({p.path} {p.start}..{p.end})")
    return w


def _realized_vol(close: np.ndarray, step_minutes: int) -> float:
    r = np.diff(np.log(close))
    per_year = 260 * 1440 / step_minutes
    return float(np.std(r) * np.sqrt(per_year)) if len(r) > 1 else 0.09


class RealReplay(Component):
    """実データの終値を、そのままの解像度で再生する。"""

    key = "source"
    stage = Stage.SOURCE
    deterministic = True

    @dataclass
    class Params(_SourceParams):
        pass

    def prepare_times(self, base: BaseParams, years, days, rng):
        s = _series(self.p, years, days)
        self._s = s
        b = replace(base, p0=float(s.close[0]), step_minutes=s.step_minutes,
                    annual_vol=_realized_vol(s.close, s.step_minutes),
                    start=str(np.datetime64(int(s.times[0]), "ns"))[:10])
        return s.times.copy(), b

    def apply(self, ctx: GenContext) -> None:
        s = self._s
        ctx.logp = np.log(s.close)
        r = np.abs(np.diff(ctx.logp, prepend=ctx.logp[0]))
        ctx.vol = np.maximum(r, 1e-12)
        ctx.meta["real_source"] = s.source
        ctx.meta["real_has_ohlc"] = s.has_ohlc


class RealSkeleton(Component):
    """実データを骨格にした 1 分足の時刻列を用意し、骨格(終値・高値・安値)を meta に置く。

    骨格の時間足は実データの解像度(日足なら日足)。block_days は bootstrap のブロック長。
    """

    key = "source"
    stage = Stage.SOURCE

    @dataclass
    class Params(_SourceParams):
        mode: str = "replay"          # replay | bootstrap
        block_days: float = 20.0

    def prepare_times(self, base: BaseParams, years, days, rng):
        s = _series(self.p, years, days)
        if s.step_minutes < base.step_minutes:
            raise ValueError("skeleton must be coarser than the base step")
        lc = np.log(s.close)
        lo_ = np.log(s.open)
        lh = np.log(np.maximum(s.high, np.maximum(s.open, s.close)))
        ll = np.log(np.minimum(s.low, np.minimum(s.open, s.close)))
        if self.p.mode == "bootstrap":
            lc, lo_, lh, ll = _block_bootstrap(lc, lo_, lh, ll, s, self.p.block_days, rng)
        elif self.p.mode != "replay":
            raise ValueError(f"unknown skeleton mode {self.p.mode}")
        start_ns = int(s.times[0]) - s.step_minutes * NS_MIN
        span_days = (int(s.times[-1]) - start_ns) / NS_DAY
        start = str(np.datetime64(start_ns, "ns"))
        times = trading_times(start, days=span_days + 1e-9, step_minutes=base.step_minutes)
        times = times[times < int(s.times[-1])]
        # 骨格の各足の終わり = その時刻より前の最後のステップ
        idx = np.searchsorted(times, s.times, side="left") - 1
        ok = idx >= 0
        self._anchor = {
            "idx": idx[ok], "close": lc[ok], "open": lo_[ok], "high": lh[ok], "low": ll[ok],
            "has_ohlc": s.has_ohlc,
        }
        b = replace(base, p0=float(np.exp(lc[0])),
                    annual_vol=_realized_vol(np.exp(lc), s.step_minutes),
                    start=start[:10])
        return times, b

    def apply(self, ctx: GenContext) -> None:
        ctx.meta["anchor"] = self._anchor
        ctx.meta["real_source"] = f"{self.p.path} ({self.p.mode})"


def _block_bootstrap(lc, lo_, lh, ll, s: RealSeries, block_days: float, rng):
    """終値のリターンと、足の形(始値・高値・安値の終値からの相対位置)を、ブロックごとに引き直す。"""
    n = len(lc)
    r = np.diff(lc)
    shape_o = (lo_ - np.r_[lc[0], lc[:-1]])[1:]     # 前の終値からの始値の窓
    shape_h = (lh - lc)[1:]
    shape_l = (ll - lc)[1:]
    bars_per_block = max(1, int(round(block_days * 1440 / s.step_minutes)))
    m = len(r)
    picks = []
    while len(picks) < m:
        st = int(rng.integers(0, max(1, m - bars_per_block)))
        picks.extend(range(st, min(m, st + bars_per_block)))
    picks = np.asarray(picks[:m])
    nr = r[picks]
    c = np.r_[lc[0], lc[0] + np.cumsum(nr)]
    o = np.r_[lo_[0], c[:-1] + shape_o[picks]]
    h = np.r_[lh[0], c[1:] + shape_h[picks]]
    lw = np.r_[ll[0], c[1:] + shape_l[picks]]
    h = np.maximum(h, np.maximum(o, c))
    lw = np.minimum(lw, np.minimum(o, c))
    assert len(c) == n
    return c, o, h, lw


class AnchorBridge(Component):
    """経路を骨格に合わせる。

    0. (calibrate_vol)合成した経路の「骨格の足 1 本分の変化」の分散が実データと同じになるよう、
       経路の振れ幅を伸縮する。指標のジャンプなどで足の中が荒れすぎるのを防ぐ
    1. 骨格の各足の終わりで、終値との差を直線補間して足し込む(ブラウン橋の補正)
    2. OHLC があれば、各足の中で max(始値, 終値) より上のはみ出しを高値に、
       min(始値, 終値) より下のはみ出しを安値に、ちょうど届くよう伸縮する
       (はみ出しが無い足は、ランダムな位置に小さな山を作って届かせる)
    """

    key = "anchor"
    stage = Stage.ANCHOR

    @dataclass
    class Params:
        fit_high_low: bool = True
        calibrate_vol: bool = True

    def apply(self, ctx: GenContext) -> None:
        a = ctx.meta.get("anchor")
        if a is None:
            return
        idx, target = a["idx"], a["close"]
        x = ctx.logp
        if self.p.calibrate_vol and len(idx) > 10:
            vx = np.var(np.diff(x[idx]))
            vt = np.var(np.diff(target))
            if vx > 0:
                k = float(np.sqrt(vt / vx))
                x = x[0] + (x - x[0]) * k
                ctx.meta["intraday_scale"] = k
        err = target - x[idx]
        steps = np.arange(ctx.n)
        x = x + np.interp(steps, idx, err)
        if a["has_ohlc"] and self.p.fit_high_low and len(idx) > 1:
            x = _fit_envelope(x, idx, a["high"], a["low"], ctx.rng)
        ctx.logp = x


def _fit_envelope(x, idx, high, low, rng):
    x = x.copy()
    for k in range(1, len(idx)):
        a, b = int(idx[k - 1]) + 1, int(idx[k]) + 1
        if b - a < 3:
            continue
        seg = x[a:b]
        o, c = x[a - 1], seg[-1]
        top, bot = max(o, c), min(o, c)
        H, L = max(high[k], top), min(low[k], bot)
        seg = seg.copy()
        inner = seg[1:-1]
        mx, mn = inner.max(), inner.min()
        if mx > top:
            up = inner > top
            inner[up] = top + (inner[up] - top) * (H - top) / (mx - top)
        elif H > top:
            j = int(rng.integers(0, len(inner)))
            inner[j] = H
        if mn < bot:
            dn = inner < bot
            inner[dn] = bot + (inner[dn] - bot) * (L - bot) / (mn - bot)
        elif L < bot:
            j = int(rng.integers(0, len(inner)))
            inner[j] = L
        seg[1:-1] = inner
        x[a:b] = seg
    return x
