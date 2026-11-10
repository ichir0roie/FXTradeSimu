"""経路をブロック(既定 15 分)単位で扱う補助。経路を見て反応する部品(POST_PATH 段)が使う。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import GenContext


def block_view(ctx: GenContext, block: int):
    """経路を block ステップごとに区切り、(開始位置, 終了位置, 高値, 安値, 終値) を返す(値は log 価格)。"""
    n = ctx.n
    starts = np.arange(0, n, block)
    ends = np.minimum(starts + block, n) - 1
    lp = ctx.logp
    return starts, ends, np.maximum.reduceat(lp, starts), np.minimum.reduceat(lp, starts), lp[ends]


def apply_block_increments(ctx: GenContext, d: np.ndarray, block: int) -> None:
    """ブロック b で決めた log 変化 d[b] を、次のブロックの中で直線的に足し込む(持続的なずれ)。"""
    n = ctx.n
    starts = np.arange(0, n, block)
    nxt = starts[1:]                    # d[b] はブロック b+1 に効く
    if len(nxt) == 0:
        return
    lens = np.minimum(nxt + block, n) - nxt
    inc = np.zeros(n)
    inc[nxt[0]:] = np.repeat(d[:len(nxt)] / lens, lens)
    ctx.logp = ctx.logp + np.cumsum(inc)


def rolling_prev_max(x: np.ndarray, window: int) -> np.ndarray:
    """x[b-window:b] の最大値(自分自身を含まない)。先頭は x[0]。"""
    out = pd.Series(x).rolling(window, min_periods=1).max().shift(1).to_numpy(copy=True)
    out[0] = x[0]
    return out


def rolling_prev_min(x: np.ndarray, window: int) -> np.ndarray:
    """x[b-window:b] の最小値(自分自身を含まない)。先頭は x[0]。"""
    out = pd.Series(x).rolling(window, min_periods=1).min().shift(1).to_numpy(copy=True)
    out[0] = x[0]
    return out
