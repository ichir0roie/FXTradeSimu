"""S009 macd_cross: MACD 線がシグナル線を上に抜けたら買い、下に抜けたら売り(ドテン、4 時間足)。

保険の損切りは ATR の倍数。12 / 26 / 9 は教科書どおりの値。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, macd


@STRATEGIES.register("macd_cross", version="1", description="MACD とシグナルの交差でドテン(12/26/9)")
class MacdCross(Strategy):
    name = "macd_cross"

    @dataclass
    class Params:
        timeframe: int = 240
        fast: int = 12
        slow: int = 26
        signal: int = 9
        atr_n: int = 14
        sl_atr: float = 3.0

    def prepare(self, bars, rng):
        self.warmup = self.p.slow + self.p.signal
        line, sig = macd(bars.close, self.p.fast, self.p.slow, self.p.signal)
        self.diff = line - sig
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)

    def on_bar(self, i, ctx):
        if i < 1:
            return
        d0, d1, a = self.diff[i - 1], self.diff[i], self.atr[i]
        if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(a)):
            return
        if d0 <= 0 < d1 and ctx.side <= 0:
            ctx.buy(sl=self.p.sl_atr * a)
        elif d0 >= 0 > d1 and ctx.side >= 0:
            ctx.sell(sl=self.p.sl_atr * a)
