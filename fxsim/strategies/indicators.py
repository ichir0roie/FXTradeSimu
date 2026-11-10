"""因果的な(未来を見ない)指標。値は位置 i までのデータだけで決まる。先頭の未定義部分は NaN。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy(copy=True)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy(copy=True)
    out[: n - 1] = np.nan
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy(copy=True)


def highest(x: np.ndarray, n: int) -> np.ndarray:
    """直近 n 本(自分を含む)の最大値。"""
    return pd.Series(x).rolling(n).max().to_numpy(copy=True)


def lowest(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).min().to_numpy(copy=True)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy(copy=True)
    out[: n - 1] = np.nan
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(copy=True)
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy(copy=True)
    out = 100 - 100 / (1 + up / np.maximum(dn, 1e-12))
    out[:n] = np.nan
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """ワイルダーの平滑(alpha = 1/n の指数平均)。"""
    out = pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy(copy=True)
    out[: n - 1] = np.nan
    return out


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """MACD 線とシグナル線。"""
    line = ema(close, fast) - ema(close, slow)
    sig = pd.Series(line).ewm(span=signal, adjust=False).mean().to_numpy(copy=True)
    sig[np.isnan(line)] = np.nan
    return line, sig


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k: int = 14,
               d: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """ストキャスティクスの %K と %D(0〜100)。"""
    hh, ll = highest(high, k), lowest(low, k)
    pk = 100 * (close - ll) / np.where(hh > ll, hh - ll, np.nan)
    return pk, sma(pk, d)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX と +DI / −DI。"""
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(high, low, close, n)
    pdi = 100 * wilder(pdm, n) / tr
    mdi = 100 * wilder(mdm, n) / tr
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    out = wilder(np.nan_to_num(dx), n)
    out[: 2 * n] = np.nan
    return out, pdi, mdi


def parabolic_sar(high: np.ndarray, low: np.ndarray, step: float = 0.02,
                  max_step: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """パラボリック SAR と向き(+1 上昇 / −1 下降)。足 i の値は足 i の高安まで使う。"""
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    if n < 2:
        return sar, trend
    up = high[1] >= high[0]
    af, ep = step, (high[1] if up else low[1])
    s = low[0] if up else high[0]
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:
                up, s, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:
                up, s, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar[i] = s
        trend[i] = 1 if up else -1
    return sar, trend


def ichimoku(high: np.ndarray, low: np.ndarray, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> dict:
    """一目均衡表。先行スパンは kijun 本先に描く線なので、足 i で見えるのは kijun 本前に計算した値。"""
    t = (highest(high, tenkan) + lowest(low, tenkan)) / 2
    k = (highest(high, kijun) + lowest(low, kijun)) / 2
    a = (t + k) / 2
    b = (highest(high, senkou) + lowest(low, senkou)) / 2
    shift = lambda x: np.r_[np.full(kijun, np.nan), x[:-kijun]] if len(x) > kijun else np.full(len(x), np.nan)  # noqa: E731
    return {"tenkan": t, "kijun": k, "span_a": shift(a), "span_b": shift(b)}


def heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: E741
    """平均足の始値と終値。"""
    hc = (o + h + l + c) / 4
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc


def zscore(x: np.ndarray, n: int) -> np.ndarray:
    m, s = sma(x, n), rolling_std(x, n)
    return (x - m) / np.where(s > 0, s, np.nan)


def previous_day_hlc(t: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray):
    """足ごとに、前の取引日(22:00 UTC 切り替え)の高値・安値・終値。最初の日は NaN。"""
    from ..calendar import trading_day

    day = trading_day(t)
    starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
    ends = np.r_[starts[1:], len(day)]
    dh = np.maximum.reduceat(high, starts)
    dl = np.minimum.reduceat(low, starts)
    dc = close[ends - 1]
    k = np.repeat(np.arange(len(starts)), ends - starts)          # 足 -> その日の番号
    ph = np.r_[np.nan, dh][k]
    pl = np.r_[np.nan, dl][k]
    pc = np.r_[np.nan, dc][k]
    return ph, pl, pc
