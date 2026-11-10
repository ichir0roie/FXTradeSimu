"""時間足(Bars)と、その組み立て。

Bars は mid の OHLC に加え bid/ask の OHLC を持つので、スプレッドの拡大で逆指値が刺さる様子も再現できる。
足の区切りは 22:00 UTC(NY クローズ)に揃える。日足 = 取引日。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calendar import NS_MIN


@dataclass
class Bars:
    """時間足。配列はすべて同じ長さ。t は足の開始時刻(int64 ns)。"""
    timeframe: int
    t: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bid_open: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    bid_close: np.ndarray
    ask_open: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    ask_close: np.ndarray
    spread_mean: np.ndarray
    spread_max: np.ndarray
    ticks: np.ndarray
    start_idx: np.ndarray      # 基準解像度での開始位置

    def __len__(self) -> int:
        return len(self.t)

    def slice(self, stop: int) -> Bars:
        """先頭から stop 本だけの Bars(先読み検査用)。"""
        kw = {k: (v[:stop] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return Bars(**kw)


BAR_OFFSET_MINUTES = 120   # 足の区切りを 22:00 UTC(NY クローズ)に揃える。日足 = 取引日


def make_bars(times: np.ndarray, mid: np.ndarray, spread: np.ndarray, minutes: int, step_minutes: int = 1) -> Bars:
    if minutes % step_minutes:
        raise ValueError(f"timeframe {minutes} must be a multiple of base step {step_minutes}")
    off = BAR_OFFSET_MINUTES * NS_MIN
    key = (times + off) // (minutes * NS_MIN)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(times)] - 1
    half = spread / 2.0
    bid = mid - half
    ask = mid + half

    def ohlc(x):
        return x[starts], np.maximum.reduceat(x, starts), np.minimum.reduceat(x, starts), x[ends]

    o, h, l, c = ohlc(mid)
    bo, bh, bl, bc = ohlc(bid)
    ao, ah, al, ac = ohlc(ask)
    cnt = (ends - starts + 1).astype(np.int64)
    return Bars(
        timeframe=minutes,
        t=key[starts] * (minutes * NS_MIN) - off,
        open=o, high=h, low=l, close=c,
        bid_open=bo, bid_high=bh, bid_low=bl, bid_close=bc,
        ask_open=ao, ask_high=ah, ask_low=al, ask_close=ac,
        spread_mean=np.add.reduceat(spread, starts) / cnt,
        spread_max=np.maximum.reduceat(spread, starts),
        ticks=cnt,
        start_idx=starts.astype(np.int64),
    )
