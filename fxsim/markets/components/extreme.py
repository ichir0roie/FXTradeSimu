"""異常な値動きと薄商い。

HolidayLiquidity  年末年始・ゴールデンウィーク・米感謝祭を「薄い」期間にする
                  (スプレッドが広がり、異常な値動きが起きやすい)
ExtremeEvents     フラッシュクラッシュ(V 字で一部戻る)、為替介入(トレンドと逆へ一気に動き、数日で一部戻る)、
                  ショック(一方向に飛んだまま戻らない)。起きている間はスプレッドが大きく広がる
FlashCrashOverlay 実データの骨格に合わせた後に重ねる、完全に戻るフラッシュクラッシュ(日足の終値を変えない)

値動きは「段」で作る(1 分の中で数十 pip 飛ぶ)。経路で約定するブローカー(intrabar="path")では、
逆指値は飛んだ先の価格で約定するので、損切りが大きく滑る。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...calendar import day_of_month, hour_of_day, trading_day, weekday
from ..base import Component, GenContext, Stage


def _month(t: np.ndarray) -> np.ndarray:
    d = t.astype("datetime64[ns]").astype("datetime64[M]")
    return (d.astype(np.int64) % 12 + 1).astype(np.int8)


class HolidayLiquidity(Component):
    """薄商いの期間(UTC の暦日)を meta["thin_mask"] に残し、ボラを少し下げる。

    - 年末年始: 12/24〜1/2
    - ゴールデンウィーク(円): 4/29〜5/5
    - 米感謝祭: 11 月の第 4 木曜とその翌日
    スプレッドの部品が thin_mask を読んで広げ、ExtremeEvents は異常な値動きの起きやすさを上げる。
    """

    key = "holidays"
    stage = Stage.VOLATILITY

    @dataclass
    class Params:
        year_end: bool = True
        golden_week: bool = True
        thanksgiving: bool = True
        vol_mult: float = 0.8

    def apply(self, ctx: GenContext) -> None:
        t = ctx.times
        m, d, dow = _month(t), day_of_month(t), weekday(t)
        thin = np.zeros(ctx.n, dtype=bool)
        if self.p.year_end:
            thin |= ((m == 12) & (d >= 24)) | ((m == 1) & (d <= 2))
        if self.p.golden_week:
            thin |= ((m == 4) & (d >= 29)) | ((m == 5) & (d <= 5))
        if self.p.thanksgiving:
            # 第 4 木曜は 22〜28 日の木曜。翌日の金曜(23〜29 日)も薄い
            thin |= (m == 11) & (((dow == 3) & (d >= 22) & (d <= 28)) | ((dow == 4) & (d >= 23) & (d <= 29)))
        ctx.vol[thin] *= self.p.vol_mult
        ctx.meta["thin_mask"] = thin


def _hour_weights(t: np.ndarray, thin: np.ndarray | None, thin_mult: float) -> np.ndarray:
    """異常な値動きが起きる場所の重み。

    ロールオーバー前後とアジア早朝(21〜23 時、0〜1 時 UTC)は 5 倍、薄い期間は thin_mult 倍。
    """
    h = hour_of_day(t)
    w = np.where((h >= 21) | (h <= 1), 5.0, 1.0)
    if thin is not None:
        w = w * np.where(thin, thin_mult, 1.0)
    return w / w.sum()


class ExtremeEvents(Component):
    """フラッシュクラッシュ・為替介入・ショック。回数は年あたりの期待値(ポアソン)。

    フラッシュクラッシュ: flash_minutes の間に flash_steps 段で落ち、recovery_hours かけて recovery_frac だけ戻る
    為替介入: 直近 trend_days の値動きと逆向きに、intervention_minutes の間に段で動き、retrace_days かけて一部戻る。
              その後 vol_days ほど荒れる
    ショック: 1 分で飛んで戻らない
    """

    key = "extreme"
    stage = Stage.POST_PATH

    @dataclass
    class Params:
        flash_per_year: float = 0.25
        flash_pct: tuple = (1.5, 5.0)
        flash_down_prob: float = 0.8           # 円高方向(USDJPY が下がる)に起きやすい
        flash_minutes: tuple = (3, 10)
        flash_steps: int = 4
        recovery_frac: tuple = (0.4, 0.8)
        recovery_hours: tuple = (0.5, 6.0)
        flash_spread_mult: float = 30.0
        intervention_per_year: float = 0.15
        intervention_pct: tuple = (1.5, 3.5)
        intervention_minutes: tuple = (20, 90)
        trend_days: float = 60.0
        retrace_frac: tuple = (0.3, 0.7)
        retrace_days: tuple = (3.0, 15.0)
        intervention_spread_mult: float = 5.0
        vol_boost: float = 2.5                 # 介入の後のボラの倍率(vol_days で減衰)
        vol_days: float = 2.0
        shock_per_year: float = 0.02
        shock_pct: tuple = (3.0, 10.0)
        shock_spread_mult: float = 50.0
        thin_rate_mult: float = 8.0            # 薄い期間の起きやすさ

    def apply(self, ctx: GenContext) -> None:
        p, rng, n = self.p, ctx.rng, ctx.n
        sm = ctx.base.step_minutes
        years = n * sm / (260 * 1440)
        weights = _hour_weights(ctx.times, ctx.meta.get("thin_mask"), p.thin_rate_mult)
        add = np.zeros(n)
        persist = np.zeros(n + 1)
        spread_mult = ctx.meta.setdefault("extreme_spread_mult", np.ones(n))
        sigma = float(np.median(ctx.vol))
        limit = self._limits(ctx)

        def place(rate: float) -> list[int]:
            k = rng.poisson(rate * years)
            return sorted(rng.choice(n, size=min(k, n), replace=False, p=weights).tolist()) if k else []

        def tail(length: int, size: float, keep: float, tau: float) -> np.ndarray:
            """size から keep * size へ時定数 tau で戻る。最後で keep * size にちょうど着く(段差を作らない)。"""
            tt = np.arange(length)
            return size * (keep + (1 - keep) * (np.exp(-tt / tau) - np.exp(-length / tau) * tt / length))

        def stairs(length: int, total: float, steps: int) -> np.ndarray:
            """length 本の間に、大きさのばらつく steps 段で total だけ動く(各段は 1 本の中で飛ぶ)。"""
            m = max(1, min(steps, length))
            edges = np.r_[0, np.sort(rng.choice(np.arange(1, length), size=m - 1, replace=False))] if m > 1 \
                else np.zeros(1, dtype=int)
            out = np.zeros(length)
            level = 0.0
            for e, frac in zip(edges, rng.dirichlet(np.ones(m)), strict=True):
                level += total * frac
                out[e:] = level
            return out

        for i in place(p.flash_per_year):
            d = -1.0 if rng.random() < p.flash_down_prob else 1.0
            size = np.log1p(rng.uniform(*p.flash_pct) / 100)
            fall = max(2, int(rng.uniform(*p.flash_minutes) / sm))
            rec = max(1.0, rng.uniform(*p.recovery_hours) * 60 / sm)
            keep = 1 - rng.uniform(*p.recovery_frac)                  # 戻らずに残る割合
            L = min(n - i, fall + int(rec * 5), limit[i] - i)
            if fall >= L:
                continue
            k = np.empty(L)
            k[:fall] = stairs(fall, size, p.flash_steps)
            k[fall:] = tail(L - fall, size, keep, rec)
            add[i:i + L] += d * k
            persist[i + L] += d * size * keep
            spread_mult[i:i + fall + int(rec / 4) + 1] *= p.flash_spread_mult
            ctx.event(i, "flash_crash", direction=int(d), pct=float(np.expm1(size) * 100),
                      pips=float(ctx.log_to_pips(size)) * d, recovered=float(1 - keep))

        logp_now = ctx.logp
        lb = max(1, int(p.trend_days * 1440 / sm))
        for i in place(p.intervention_per_year):
            trend = logp_now[i] - logp_now[max(0, i - lb)]
            d = -1.0 if trend > 0 else 1.0                            # 直近の動きと逆へ
            size = np.log1p(rng.uniform(*p.intervention_pct) / 100)
            dur = max(2, int(rng.uniform(*p.intervention_minutes) / sm))
            ret_steps = max(1.0, rng.uniform(*p.retrace_days) * 1440 / sm)
            keep = 1 - rng.uniform(*p.retrace_frac)
            L = min(n - i, dur + int(ret_steps * 3), limit[i] - i)
            if dur >= L:
                continue
            k = np.empty(L)
            k[:dur] = stairs(dur, size, max(3, dur // 5))
            k[dur:] = tail(L - dur, size, keep, ret_steps)
            # 介入の後はしばらく荒れる(上乗せの揺れ。持続する)
            vlen = min(n - i - dur, int(p.vol_days * 1440 / sm * 3))
            if vlen > 0:
                decay = np.exp(-np.arange(vlen) / (p.vol_days * 1440 / sm))
                noise = rng.standard_normal(vlen) * sigma * (p.vol_boost - 1) * decay
                persist[i + dur:i + dur + vlen] += noise
            add[i:i + L] += d * k
            persist[i + L] += d * size * keep
            spread_mult[i:i + dur * 2] *= p.intervention_spread_mult
            ctx.event(i, "intervention", direction=int(d), pct=float(np.expm1(size) * 100),
                      pips=float(ctx.log_to_pips(size)) * d)

        for i in place(p.shock_per_year):
            d = -1.0 if rng.random() < 0.5 else 1.0
            size = np.log1p(rng.uniform(*p.shock_pct) / 100)
            persist[i] += d * size
            spread_mult[i:i + max(1, 60 // sm)] *= p.shock_spread_mult
            ctx.event(i, "shock", direction=int(d), pct=float(np.expm1(size) * 100),
                      pips=float(ctx.log_to_pips(size)) * d)

        ctx.logp = ctx.logp + add + np.cumsum(persist[:n])

    def _limits(self, ctx: GenContext) -> np.ndarray:
        """足 i で始まった動きが届いてよい位置(この位置の手前で終える)。既定は系列の終わり。"""
        return np.full(ctx.n, ctx.n)


class FlashCrashOverlay(ExtremeEvents):
    """実データの骨格に合わせた後(ANCHOR 段)に重ねる、完全に戻るフラッシュクラッシュだけの版。

    その取引日の終値と骨格の足の終値(meta["anchor"]["idx"])の手前で必ず戻り切るので、日足の終値は実データのまま
    (高値・安値は実データを越えることがある)。終値の直前に起きて戻る余地が無いものは起こさない。
    介入やショックは実データの日足にすでに入っているので起こさない。
    AnchorBridge より後に並べること(同じ ANCHOR 段なので、並べた順に動く)。
    """

    key = "flash_overlay"
    stage = Stage.ANCHOR

    @dataclass
    class Params(ExtremeEvents.Params):
        recovery_frac: tuple = (1.0, 1.0)
        recovery_hours: tuple = (0.25, 3.0)
        intervention_per_year: float = 0.0
        shock_per_year: float = 0.0

    def _limits(self, ctx: GenContext) -> np.ndarray:
        """その取引日の最後の足(日足の終値)と、骨格の終値の位置の手前で終える。"""
        pos = np.arange(ctx.n)
        day = trading_day(ctx.times)
        last = np.r_[np.flatnonzero(np.diff(day) != 0), ctx.n - 1]       # 各取引日の最後の足
        lim = last[np.searchsorted(last, pos, side="left")]
        a = ctx.meta.get("anchor")
        if a is not None and len(a["idx"]):
            idx = np.asarray(a["idx"])
            j = np.searchsorted(idx, pos, side="left")                   # i 以降で最初の骨格の終値
            lim = np.minimum(lim, np.where(j < len(idx), idx[np.minimum(j, len(idx) - 1)], ctx.n))
        return lim


__all__ = ["ExtremeEvents", "FlashCrashOverlay", "HolidayLiquidity"]
