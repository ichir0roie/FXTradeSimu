"""S052 main_month_end_fix: 月末のロンドン 16 時のフィキシングの後に、ドルを買って数時間で出る(E061)。

E061 の研究の約定では、月の最後の平日の 16:00 ロンドン(夏時間 15:00 UTC、冬 16:00 UTC)から 1〜2 時間、対ドル 6 ペアで
ドルが買い戻される(素の +2〜4 bp、t 3)。ここではそれを本物のエンジン(MT5 の決まり、記録のスプレッド)で確かめる。
1 時間足で動かす。足 i の終値で決める向き target[i] は、次の足の始値で効く:
    フィキシングの直前の足(始まりが 16 時ロンドンの 1 時間前)から hold 本、ドルを買う向き(USDxxx は買い、xxxUSD は売り)
    その後は 0(時間で出る)。ドルを含まないペアは取引しない
損切りの幅は ATR(24 本) x stop_atr(量を決めるだけで、数時間で出るのでほとんど届かない)。
費用のゲートは使わない(1 回の見込みが 1〜2 pip で、ゲートの 3 倍の基準に届かない。判定は帰無との差で見る)。
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .. import STRATEGIES
from ..indicators import atr
from .base import MainStrategy

LONDON = ZoneInfo("Europe/London")
NS = 1_000_000_000


def usd_side(pair: str | None) -> int:
    """ドルを買う向き(USD が基軸なら +1、決済なら −1、無ければ 0)。"""
    if not pair:
        return 0
    p = pair.replace("/", "").upper()
    return 1 if p.startswith("USD") else -1 if p.endswith("USD") else 0


def fix_bar_starts(t_ns: np.ndarray) -> set[int]:
    """月の最後の平日の、16:00 ロンドンの 1 時間前に始まる 1 時間足の始まり(UTC の秒)。"""
    days = {datetime.fromtimestamp(int(x) // NS, timezone.utc).date() for x in t_ns[::24]}
    days |= {datetime.fromtimestamp(int(t_ns[-1]) // NS, timezone.utc).date()}
    months = {(d.year, d.month) for d in days}
    out = set()
    for y, m in months:
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        fix = datetime(last.year, last.month, last.day, 16, tzinfo=LONDON).astimezone(timezone.utc)
        out.add(int(fix.timestamp()) - 3600)
    return out


@STRATEGIES.register("main_month_end_fix", version="1",
                     description="本命: 月末のロンドン 16 時の後にドルを買い、数時間で出る(E061、1 時間足)")
class MainMonthEndFix(MainStrategy):
    name = "main_month_end_fix"

    @dataclass
    class Params:
        timeframe: int = 60
        guard: dict = field(default_factory=lambda: {"cost_multiple": 0.0, "hold_days_est": 0.1,
                                                     "max_trades_per_year": 24.0})
        hold: int = 2
        atr_n: int = 24
        stop_atr: float = 3.0

    def signals(self, bars, rng):
        n = len(bars)
        side = usd_side(getattr(bars, "pair", None))
        target = np.zeros(n, dtype=int)
        if side != 0 and n:
            starts = fix_bar_starts(bars.t)
            sec = bars.t // NS
            for i in np.flatnonzero(np.isin(sec, list(starts))):
                target[i:i + self.p.hold] = side
        a = atr(bars.high, bars.low, bars.close, self.p.atr_n)
        self.warmup = self.p.atr_n
        return target, self.p.stop_atr * a, a * np.sqrt(self.p.hold)
