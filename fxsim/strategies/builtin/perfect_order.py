"""S017 perfect_order: パーフェクトオーダー(4 時間足)。

短期・中期・長期の移動平均が上から順に並び(5 > 20 > 60)、3 本とも上向きなら買い。逆なら売り。並びが崩れたら手仕舞い。
日本の個人投資家に人気の順張りの型。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, sma


@STRATEGIES.register("perfect_order", version="1", description="移動平均 3 本が順に並び、すべて同じ向きのときに乗る")
class PerfectOrder(Strategy):
    name = "perfect_order"

    @dataclass
    class Params:
        timeframe: int = 240
        short: int = 5
        mid: int = 20
        long: int = 60
        atr_n: int = 14
        sl_atr: float = 3.0

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = p.long + 2
        c = bars.close
        s, m, lg = sma(c, p.short), sma(c, p.mid), sma(c, p.long)
        rising = lambda x: np.r_[False, x[1:] > x[:-1]]   # noqa: E731
        falling = lambda x: np.r_[False, x[1:] < x[:-1]]  # noqa: E731
        self.up = (s > m) & (m > lg) & rising(s) & rising(m) & rising(lg)
        self.dn = (s < m) & (m < lg) & falling(s) & falling(m) & falling(lg)
        self.order_up = (s > m) & (m > lg)
        self.order_dn = (s < m) & (m < lg)
        self.atr = atr(bars.high, bars.low, c, p.atr_n)

    def on_bar(self, i, ctx):
        a = self.atr[i]
        if not np.isfinite(a):
            return
        if ctx.side > 0 and not self.order_up[i] or ctx.side < 0 and not self.order_dn[i]:
            ctx.close()
        elif ctx.side == 0:
            if self.up[i]:
                ctx.buy(sl=self.p.sl_atr * a)
            elif self.dn[i]:
                ctx.sell(sl=self.p.sl_atr * a)
