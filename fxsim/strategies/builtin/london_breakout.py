"""S007 london_breakout: 東京時間のレンジを、ロンドンの寄り付きで抜けた方へ(デイトレード、15 分足)。

レンジ = 23:00〜06:00 UTC の高値・安値。07:00〜10:00 UTC に終値でレンジを抜けたら、抜けた方へ入る。
損切りはレンジの反対側(長すぎるときは max_sl_pips)、利確はレンジ幅の tp_range 倍。16:00 UTC(ロンドンの引け)で手仕舞う。
1 日 1 回まで。ダマシとストップ狩りの直撃を受ける典型的なデイトレード。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy


@STRATEGIES.register("london_breakout", version="1",
                     description="東京時間のレンジをロンドンの寄りで抜けた方へ(デイトレ)")
class LondonBreakout(Strategy):
    name = "london_breakout"

    @dataclass
    class Params:
        timeframe: int = 15
        range_start: int = 23          # UTC の時(前日)
        range_end: int = 6             # この時の終わりまで
        entry_start: int = 7
        entry_end: int = 10
        exit_hour: int = 16
        buffer_pips: float = 2.0
        tp_range: float = 1.0
        max_sl_pips: float = 40.0
        min_range_pips: float = 10.0

    def prepare(self, bars, rng):
        p = self.p
        # 足の終わりの時刻(この時刻の終値で判断する)
        tc = bars.t + p.timeframe * 60_000_000_000
        h = ((tc // 3_600_000_000_000) % 24).astype(int)
        m = ((tc // 60_000_000_000) % 60).astype(int)
        hour = h + m / 60
        in_range = (hour > p.range_start) | (hour <= p.range_end + 1e-9) if p.range_start > p.range_end \
            else (hour > p.range_start) & (hour <= p.range_end + 1e-9)
        n = len(bars)
        self.hi = np.full(n, np.nan)
        self.lo = np.full(n, np.nan)
        hi = lo = np.nan
        prev_in = False
        for i in range(n):                      # レンジの間だけ高安を更新し、終わったら翌日まで固定
            if in_range[i]:
                if not prev_in:
                    hi, lo = bars.high[i], bars.low[i]
                else:
                    hi, lo = max(hi, bars.high[i]), min(lo, bars.low[i])
            prev_in = in_range[i]
            self.hi[i], self.lo[i] = hi, lo
        self.hour = hour
        self.c = bars.close
        self.day = (tc + 2 * 3_600_000_000_000) // 86_400_000_000_000
        self.traded_day = -1

    def on_bar(self, i, ctx):
        p, h = self.p, self.hour[i]
        if ctx.side != 0:
            if not (p.entry_start <= h < p.exit_hour):       # ロンドンの引けを過ぎたら持ち越さない
                ctx.close()
            return
        if not (p.entry_start <= h <= p.entry_end) or self.day[i] == self.traded_day:
            return
        hi, lo, c, pip = self.hi[i], self.lo[i], self.c[i], ctx.pip
        if not (np.isfinite(hi) and np.isfinite(lo)) or (hi - lo) < p.min_range_pips * pip:
            return
        width = hi - lo
        if c > hi + p.buffer_pips * pip:
            ctx.buy(sl=min(c - lo, p.max_sl_pips * pip), tp=p.tp_range * width)
            self.traded_day = self.day[i]
        elif c < lo - p.buffer_pips * pip:
            ctx.sell(sl=min(hi - c, p.max_sl_pips * pip), tp=p.tp_range * width)
            self.traded_day = self.day[i]
