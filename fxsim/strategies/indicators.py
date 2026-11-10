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
