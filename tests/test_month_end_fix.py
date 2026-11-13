from datetime import datetime, timezone

import numpy as np

from fxsim.strategies.main.month_end_fix import NS, fix_bar_starts, usd_side


def _ts(y, m, d, h):
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp())


def test_usd_side():
    assert usd_side("USDJPY") == 1
    assert usd_side("EUR/USD") == -1
    assert usd_side("EURJPY") == 0
    assert usd_side(None) == 0


def test_fix_bar_is_the_hour_before_4pm_london_on_the_last_weekday():
    t = np.arange(_ts(2024, 1, 1, 0), _ts(2024, 12, 31, 23), 3600, dtype=np.int64) * NS
    starts = fix_bar_starts(t)
    # 2024-03-29 は金曜(月の最後の平日)、ロンドンは冬時間 → 16:00 UTC、その前の足は 15:00 UTC
    assert _ts(2024, 3, 29, 15) in starts
    # 2024-06-28 は金曜、夏時間 → 15:00 UTC、その前の足は 14:00 UTC
    assert _ts(2024, 6, 28, 14) in starts
    # 2024-08-31 は土曜 → 30 日(金)
    assert _ts(2024, 8, 30, 14) in starts
    assert len([s for s in starts if _ts(2024, 1, 1, 0) <= s <= _ts(2024, 12, 31, 23)]) == 12
