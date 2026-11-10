"""S014 ichimoku: 一目均衡表(4 時間足)。

転換線が基準線を上に抜け、終値が雲(先行スパン A・B)の上なら買い。逆なら売り。
反対の交差か、終値が雲の中に入ったら手仕舞い。保険の損切りは ATR の倍数。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, ichimoku


@STRATEGIES.register("ichimoku", version="1", description="一目均衡表: 転換線と基準線の交差 + 雲の上下")
class Ichimoku(Strategy):
    name = "ichimoku"

    @dataclass
    class Params:
        timeframe: int = 240
        tenkan: int = 9
        kijun: int = 26
        senkou: int = 52
        atr_n: int = 14
        sl_atr: float = 3.0

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = p.senkou + p.kijun
        x = ichimoku(bars.high, bars.low, p.tenkan, p.kijun, p.senkou)
        self.d = x["tenkan"] - x["kijun"]
        self.top = np.fmax(x["span_a"], x["span_b"])
        self.bot = np.fmin(x["span_a"], x["span_b"])
        self.c = bars.close
        self.atr = atr(bars.high, bars.low, bars.close, p.atr_n)

    def on_bar(self, i, ctx):
        if i < 1:
            return
        d0, d1, c, top, bot, a = self.d[i - 1], self.d[i], self.c[i], self.top[i], self.bot[i], self.atr[i]
        if not np.all(np.isfinite([d0, d1, top, bot, a])):
            return
        if ctx.side > 0 and (d1 < 0 or c < top) or ctx.side < 0 and (d1 > 0 or c > bot):
            ctx.close()
        elif ctx.side == 0:
            if d0 <= 0 < d1 and c > top:
                ctx.buy(sl=self.p.sl_atr * a)
            elif d0 >= 0 > d1 and c < bot:
                ctx.sell(sl=self.p.sl_atr * a)
