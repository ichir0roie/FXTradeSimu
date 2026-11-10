"""S016 adx_trend: ADX でトレンドの強さを測り、強いときだけ +DI / −DI の向きへ(4 時間足)。

ADX > 25 かつ +DI > −DI で買い、逆で売り。ADX < 20 に落ちるか DI が逆転したら手仕舞い。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import adx, atr


@STRATEGIES.register("adx_trend", version="1", description="ADX > 25 のときだけ DI の向きへ(トレンドの強さで絞る)")
class AdxTrend(Strategy):
    name = "adx_trend"

    @dataclass
    class Params:
        timeframe: int = 240
        n: int = 14
        enter: float = 25.0
        leave: float = 20.0
        sl_atr: float = 3.0

    def prepare(self, bars, rng):
        self.warmup = 3 * self.p.n
        self.adx, self.pdi, self.mdi = adx(bars.high, bars.low, bars.close, self.p.n)
        self.atr = atr(bars.high, bars.low, bars.close, self.p.n)

    def on_bar(self, i, ctx):
        x, pd_, md, a = self.adx[i], self.pdi[i], self.mdi[i], self.atr[i]
        if not np.all(np.isfinite([x, pd_, md, a])):
            return
        if ctx.side > 0 and (x < self.p.leave or pd_ < md) or ctx.side < 0 and (x < self.p.leave or md < pd_):
            ctx.close()
        elif ctx.side == 0 and x > self.p.enter:
            if pd_ > md:
                ctx.buy(sl=self.p.sl_atr * a)
            else:
                ctx.sell(sl=self.p.sl_atr * a)
