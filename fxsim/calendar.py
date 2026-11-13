"""FX の取引カレンダー(UTC)。

- 週は日曜 22:00 UTC に始まり、金曜 22:00 UTC に終わる(土曜は休み)。
- 日付の切り替え(ロールオーバー)は 22:00 UTC。スワップはこの時刻に付く。
- 祝日は扱わない。
時刻はすべて int64 のナノ秒(numpy の datetime64[ns] と同じ値)で持つ。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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


def parse_server_tz(rule: str) -> tuple[str, int]:
    """'America/New_York+7' -> ('America/New_York', 7)。サーバー時刻 = その地域の時刻 + 時間。'UTC+2' は固定のずれ。"""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    rule = rule.strip()
    try:                       # Etc/GMT-2 のように、符号まで含めて地域の名前のもの
        ZoneInfo(rule)
        return rule, 0
    except (ZoneInfoNotFoundError, ValueError):
        pass
    for sign in ("+", "-"):
        zone, sep, hours = rule.rpartition(sign)
        if sep and zone and hours.replace(".", "", 1).isdigit():
            return zone, round(float(hours) * 3600) * (1 if sign == "+" else -1)
    return rule, 0


def server_hour(t_ns: np.ndarray, rule: str) -> np.ndarray:
    """UTC の時刻(ns)の、MT5 のサーバー時刻での時(0〜23)。rule は parse_server_tz の書式。

    OANDA Japan など「NY + 7 時間」の業者では、ロールオーバー(NY の 17 時)が夏も冬もサーバーの 0 時になる。
    """
    import pandas as pd

    zone, shift = parse_server_tz(rule)
    local = pd.to_datetime(np.asarray(t_ns, dtype=np.int64), utc=True).tz_convert(zone)
    sec = local.hour.to_numpy() * 3600 + local.minute.to_numpy() * 60 + local.second.to_numpy() + shift
    return (sec // 3600) % 24


# ---- 米雇用統計(E062)----
# BLS の決まり: 対象月の 12 日を含む週(日〜土)が終わった土曜の 20 日後の金曜、8:30 ニューヨーク。
# 1 月 3 日以前なら 1 週後、7 月 4 日・3 日(振替)の金曜なら前の木曜。
# 政府の閉鎖でずれた回は覚えている公表日に置き換えた。
NFP_OVERRIDES = {(2013, 9): date(2013, 10, 22), (2013, 10): date(2013, 11, 8),
                 (2025, 9): date(2025, 11, 20), (2025, 10): None, (2025, 11): date(2025, 12, 16)}


def nfp_dates(first: date, last: date) -> list[date]:
    out = []
    y, m = first.year, first.month
    while True:
        if (y, m) in NFP_OVERRIDES:
            d = NFP_OVERRIDES[(y, m)]
        else:
            twelfth = date(y, m, 12)
            sat = twelfth + timedelta(days=(5 - twelfth.weekday()) % 7)
            d = sat + timedelta(days=20)
            if d.month == 1 and d.day <= 3:          # 年始は 1 週後ろ(2014・2015・2020・2025・2026 など)
                d += timedelta(days=7)
            elif d.month == 7 and d.day in (3, 4):
                d -= timedelta(days=1)
        if d is not None:
            if d > last:
                break
            if d >= first:
                out.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def nfp_bar_start(d: date) -> int:
    """8:30 ニューヨークを含む 1 時間足の始まり(UTC の秒)。"""
    t = datetime(d.year, d.month, d.day, 8, 30, tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
    return int(t.replace(minute=0).timestamp())


# ---- 米新規失業保険申請件数(E067)----
# 毎週木曜 8:30 ニューヨーク。木曜が連邦の祝日(元日・独立記念日・ジューンティーンス(2021〜)・復員軍人の日・
# 感謝祭・クリスマス)なら前の水曜。
def _thanksgiving(y: int) -> date:
    d = date(y, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(days=21)


def claims_dates(first: date, last: date) -> list[date]:
    out = []
    d = first + timedelta(days=(3 - first.weekday()) % 7)
    while d <= last:
        hol = {date(d.year, 1, 1), date(d.year, 7, 4), date(d.year, 11, 11), date(d.year, 12, 25),
               _thanksgiving(d.year)}
        if d.year >= 2021:
            hol.add(date(d.year, 6, 19))
        out.append(d - timedelta(days=1) if d in hol else d)
        d += timedelta(days=7)
    return out
