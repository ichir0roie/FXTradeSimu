"""S018 heikin_ashi: 平均足の色が confirm 本続けて変わったら、その向きへドテン(4 時間足)。"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, heikin_ashi


@STRATEGIES.register("heikin_ashi", version="1", description="平均足の色が続けて変わったらドテン")
class HeikinAshi(Strategy):
    name = "heikin_ashi"

    @dataclass
    class Params:
        timeframe: int = 240
        confirm: int = 2
        atr_n: int = 14
        sl_atr: float = 3.0

    def prepare(self, bars, rng):
        self.warmup = self.p.confirm + self.p.atr_n
        ho, hc = heikin_ashi(bars.open, bars.high, bars.low, bars.close)
        color = np.sign(hc - ho)
        k = self.p.confirm
        self.want = np.zeros(len(color))
        for j in range(k - 1, len(color)):
            w = color[j - k + 1:j + 1]
            if np.all(w > 0):
                self.want[j] = 1
            elif np.all(w < 0):
                self.want[j] = -1
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)

    def on_bar(self, i, ctx):
        w, a = int(self.want[i]), self.atr[i]
        if w == 0 or w == ctx.side or not np.isfinite(a):
            return
        if w > 0:
            ctx.buy(sl=self.p.sl_atr * a)
        else:
            ctx.sell(sl=self.p.sl_atr * a)
