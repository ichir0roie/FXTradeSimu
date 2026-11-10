"""動作ケースの「いつ判断し、いつ取引してよいか」。

TradingSchedule は足の時刻から 3 つの真偽の列を作る:
    decide[i]       足 i の終値で戦略に判断させるか(判断の間隔)
    can_enter[i]    足 i の終値で新規注文を出してよいか(取引時間)
    force_close[i]  足 i の終値で持ち高を手仕舞うか(取引時間外は持たない設定のとき)
決済(close)と損切り・利確は、取引時間外でも常に有効。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..calendar import NS_MIN, weekday

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TZ_OFFSETS = {"UTC": 0, "JST": 9 * 60, "LON": 0, "NY": -5 * 60}   # 夏時間は扱わない(固定の時差)


def _minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass
class TradingSchedule:
    decision_every: int = 1                 # 何本ごとに判断するか(1 = 毎本)
    hours: list | None = None               # 取引してよい時間帯 [["07:00", "16:00"], ...](tz の時刻、終わりは含まない)
    tz: str = "UTC"                         # hours / days の時刻の基準(UTC / JST / LON / NY、固定の時差)
    days: list = field(default_factory=lambda: list(DAY_NAMES[:5]))   # 取引してよい曜日
    outside: str = "hold"                   # 時間外の持ち高: hold(持ち越す)/ close(手仕舞う)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TradingSchedule":
        return cls(**(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)

    def masks(self, t: np.ndarray, timeframe: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(t)
        if self.tz not in TZ_OFFSETS:
            raise ValueError(f"unknown tz {self.tz}. known: {sorted(TZ_OFFSETS)}")
        if self.outside not in ("hold", "close"):
            raise ValueError("outside must be 'hold' or 'close'")
        # 判断は足の終値 = 足の終わりの時刻で行う
        close_t = t + timeframe * NS_MIN + TZ_OFFSETS[self.tz] * NS_MIN
        mod = (close_t // NS_MIN) % 1440
        ok_day = np.isin(weekday(close_t - NS_MIN), [DAY_NAMES.index(d) for d in self.days])
        if self.hours:
            ok_hour = np.zeros(n, dtype=bool)
            for a, b in self.hours:
                a, b = _minutes(a), _minutes(b)
                ok_hour |= ((mod >= a) & (mod < b)) if a < b else ((mod >= a) | (mod < b))
        else:
            ok_hour = np.ones(n, dtype=bool)
        can_enter = ok_day & ok_hour
        decide = (np.arange(n) % max(1, int(self.decision_every))) == 0
        force_close = ~can_enter if self.outside == "close" else np.zeros(n, dtype=bool)
        return decide, can_enter, force_close
