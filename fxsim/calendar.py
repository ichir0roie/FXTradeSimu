"""FX の取引カレンダー(UTC)。

- 週は日曜 22:00 UTC に始まり、金曜 22:00 UTC に終わる(土曜は休み)。
- 日付の切り替え(ロールオーバー)は 22:00 UTC。スワップはこの時刻に付く。
- 祝日は扱わない。
時刻はすべて int64 のナノ秒(numpy の datetime64[ns] と同じ値)で持つ。
"""
from __future__ import annotations

import numpy as np

NS_MIN = 60_000_000_000
NS_HOUR = 60 * NS_MIN
NS_DAY = 24 * NS_HOUR
TRADING_MINUTES_PER_YEAR = 260 * 1440


def trading_times(start: str = "2015-01-05", *, days: float | None = None, years: float | None = None,
                  step_minutes: int = 1) -> np.ndarray:
    """取引時間中の時刻列(int64 ns)。days は暦日、years は 365.25 暦日として数える。"""
    if (days is None) == (years is None):
        raise ValueError("specify exactly one of days / years")
    total_days = days if days is not None else years * 365.25
    start_ns = np.datetime64(start, "ns").astype(np.int64)
    n_all = int(total_days * 1440 / step_minutes)
    t = start_ns + np.arange(n_all, dtype=np.int64) * (step_minutes * NS_MIN)
    dow = weekday(t)
    hour = hour_of_day(t)
    keep = (dow <= 3) | ((dow == 4) & (hour < 22)) | ((dow == 6) & (hour >= 22))
    return t[keep]


def hour_of_day(t: np.ndarray) -> np.ndarray:
    return ((t // NS_HOUR) % 24).astype(np.int16)


def minute_of_day(t: np.ndarray) -> np.ndarray:
    return ((t // NS_MIN) % 1440).astype(np.int16)


def weekday(t: np.ndarray) -> np.ndarray:
    """月曜=0 ... 日曜=6(1970-01-01 は木曜)。"""
    return (((t // NS_DAY) + 3) % 7).astype(np.int8)


def day_of_month(t: np.ndarray) -> np.ndarray:
    d = t.astype("datetime64[ns]").astype("datetime64[D]")
    return (d - d.astype("datetime64[M]")).astype(np.int64) + 1


def trading_day(t: np.ndarray | int) -> np.ndarray | int:
    """22:00 UTC で切り替わる取引日の通し番号(1970-01-01 起点の日数)。"""
    return (np.asarray(t) + 2 * NS_HOUR) // NS_DAY


def rollover_weight(day_id: int) -> int:
    """取引日 day_id の終わりに付くスワップの日数。水曜は週末分を含めて 3、土日は 0。"""
    dow = (day_id + 3) % 7
    if dow == 2:
        return 3
    if dow >= 5:
        return 0
    return 1


def steps_per_year(step_minutes: int) -> float:
    return TRADING_MINUTES_PER_YEAR / step_minutes
