"""MT5 の足(copy_rates_*)を Bars にする。

MT5 の足は bid の OHLC と、ポイント単位のスプレッドを持つ。ask = bid + スプレッド、mid = bid + スプレッド / 2 とする
(足の中のスプレッドの動きは分からないので、足ごとに一定とみなす)。
時刻は MT5 のサーバー時刻(秒)なので、サーバー時刻 − UTC を引いて UTC の ns にする。
足の中の経路は無いので、損切り・利確は足の高安で判定される(両方届いたら損切りが先)。
"""
from __future__ import annotations

import numpy as np

from ..data import Bars

# MT5 の時間足の名前(TIMEFRAME_*)。動作ケースの timeframe(分)から引く
TIMEFRAMES = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1"}


def rates_to_bars(rates: np.ndarray, timeframe: int, point: float, utc_offset_sec: int) -> Bars:
    """rates は MT5 の構造化配列(time, open, high, low, close, tick_volume, spread, ...)。"""
    t = (np.asarray(rates["time"], dtype=np.int64) - int(utc_offset_sec)) * 1_000_000_000
    bo = np.asarray(rates["open"], dtype=float)
    bh = np.asarray(rates["high"], dtype=float)
    bl = np.asarray(rates["low"], dtype=float)
    bc = np.asarray(rates["close"], dtype=float)
    s = np.asarray(rates["spread"], dtype=float) * point
    ticks = np.asarray(rates["tick_volume"], dtype=np.int64)
    half = s / 2
    n = len(t)
    return Bars(timeframe=timeframe, t=t,
                open=bo + half, high=bh + half, low=bl + half, close=bc + half,
                bid_open=bo, bid_high=bh, bid_low=bl, bid_close=bc,
                ask_open=bo + s, ask_high=bh + s, ask_low=bl + s, ask_close=bc + s,
                spread_mean=s, spread_max=s.copy(), ticks=ticks, start_idx=np.arange(n, dtype=np.int64))


def pip_from_digits(digits: int) -> float:
    """3 桁・2 桁(対円)なら 0.01、5 桁・4 桁なら 0.0001。"""
    return 0.01 if digits <= 3 else 0.0001
