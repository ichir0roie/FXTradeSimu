"""数値計算の小物。scipy があれば使い、無ければ numpy だけで同じ結果を出す。"""
from __future__ import annotations

import numpy as np


def ar1_filter(e: np.ndarray, a: float, x0: float = 0.0) -> np.ndarray:
    """x_t = a * x_{t-1} + e_t を計算する(x_{-1} = x0)。"""
    e = np.asarray(e, dtype=np.float64)
    if a == 0.0:
        return e.copy()
    try:
        from scipy.signal import lfilter

        y, _ = lfilter([1.0], [1.0, -a], e, zi=np.array([a * x0]))
        return y
    except ImportError:  # pragma: no cover - scipy が無い環境向け
        return _ar1_chunked(e, a, x0)


def _ar1_chunked(e: np.ndarray, a: float, x0: float) -> np.ndarray:
    """ブロックごとに x_j = a^j (a x_prev + sum_k e_k a^{-k}) を使ってベクトル化する。"""
    n = len(e)
    out = np.empty(n)
    la = -np.log(abs(a))
    block = int(max(1, min(8192, 30.0 / la))) if la > 0 else 8192
    prev = x0
    for s in range(0, n, block):
        seg = e[s:s + block]
        j = np.arange(len(seg))
        pw = a ** j
        acc = np.cumsum(seg / pw)
        x = pw * (a * prev + acc)
        out[s:s + len(seg)] = x
        prev = x[-1]
    return out


def ema(x: np.ndarray, halflife: float) -> np.ndarray:
    """半減期 halflife(要素数)の指数移動平均。因果的(未来を見ない)。"""
    a = 0.5 ** (1.0 / max(halflife, 1e-9))
    x = np.asarray(x, dtype=np.float64)
    return ar1_filter((1 - a) * x, a, x0=x[0])


def student_t_unit(rng: np.random.Generator, df: float, size) -> np.ndarray:
    """分散 1 に正規化した t 分布の乱数(df > 2)。"""
    z = rng.standard_t(df, size=size)
    return z * np.sqrt((df - 2.0) / df)
