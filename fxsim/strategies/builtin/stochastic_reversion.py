"""S010 stochastic_reversion: ストキャスティクスの %K が %D を、売られすぎ(20 未満)で上に抜けたら買い、
買われすぎ(80 超)で下に抜けたら売り(1 時間足)。反対の水準か 50 で手仕舞い。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, stochastic


@STRATEGIES.register("stochastic_reversion", version="1",
                     description="ストキャスの売られすぎ・買われすぎでの交差に逆張り")
class StochasticReversion(Strategy):
    name = "stochastic_reversion"

    @dataclass
    class Params:
        timeframe: int = 60
        k: int = 14
        d: int = 3
        low: float = 20.0
        high: float = 80.0
        atr_n: int = 24
        sl_atr: float = 2.5
        max_hold: int = 48

    def prepare(self, bars, rng):
        self.warmup = self.p.k + self.p.d + 1
        self.pk, self.pd = stochastic(bars.high, bars.low, bars.close, self.p.k, self.p.d)
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)

    def on_bar(self, i, ctx):
        if i < 1:
            return
        p = self.p
        k0, d0, k1, d1, a = self.pk[i - 1], self.pd[i - 1], self.pk[i], self.pd[i], self.atr[i]
        if not np.all(np.isfinite([k0, d0, k1, d1, a])):
            return
        if ctx.side != 0:
            held = i - ctx.position.entry_idx
            if (ctx.side > 0 and k1 >= 50) or (ctx.side < 0 and k1 <= 50) or held >= p.max_hold:
                ctx.close()
            return
        if k0 <= d0 and k1 > d1 and k1 < p.low:
            ctx.buy(sl=p.sl_atr * a)
        elif k0 >= d0 and k1 < d1 and k1 > p.high:
            ctx.sell(sl=p.sl_atr * a)
