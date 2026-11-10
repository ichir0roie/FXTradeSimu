"""S002 donchian_breakout: 直近 N 本の高値/安値を終値で抜けたら入る(タートル風)。

出口は反対側の M 本チャネル、または ATR 基準の損切り。ダマシ・ストップ狩りの直撃を受ける型。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, highest, lowest


@STRATEGIES.register("donchian_breakout", version="1", description="N 本チャネルのブレイクアウト(順張り)")
class DonchianBreakout(Strategy):
    name = "donchian_breakout"

    @dataclass
    class Params:
        timeframe: int = 60
        entry_n: int = 48
        exit_n: int = 24
        atr_n: int = 24
        sl_atr: float = 2.0

    def prepare(self, bars, rng):
        self.warmup = self.p.entry_n + 1
        c = bars.close
        # 自分を含まない直近 N 本(i-N .. i-1)
        self.hi = np.r_[np.nan, highest(bars.high, self.p.entry_n)[:-1]]
        self.lo = np.r_[np.nan, lowest(bars.low, self.p.entry_n)[:-1]]
        self.xhi = np.r_[np.nan, highest(bars.high, self.p.exit_n)[:-1]]
        self.xlo = np.r_[np.nan, lowest(bars.low, self.p.exit_n)[:-1]]
        self.c = c
        self.atr = atr(bars.high, bars.low, c, self.p.atr_n)

    def on_bar(self, i, ctx):
        c, a = self.c[i], self.atr[i]
        if not np.isfinite(a) or not np.isfinite(self.hi[i]):
            return
        side = ctx.side
        if side > 0 and c < self.xlo[i]:
            ctx.close()
            return
        if side < 0 and c > self.xhi[i]:
            ctx.close()
            return
        if side == 0:
            if c > self.hi[i]:
                ctx.buy(sl=self.p.sl_atr * a)
            elif c < self.lo[i]:
                ctx.sell(sl=self.p.sl_atr * a)
