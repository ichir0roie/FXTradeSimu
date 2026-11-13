"""S012 pivot_reversal: 前日の高安終値から引くピボットで逆張り(デイトレード、15 分足)。

P = (H + L + C) / 3、R1 = 2P − L、S1 = 2P − H、R2 = P + (H − L)、S2 = P − (H − L)。
終値が R1 に届いたら売り(利確 P、損切り R2)、S1 に届いたら買い(利確 P、損切り S2)。1 日に各向き 1 回まで。
"""
from dataclasses import dataclass

import numpy as np

from ...calendar import trading_day
from .. import STRATEGIES
from ..base import Strategy
from ..indicators import previous_day_hlc


@STRATEGIES.register("pivot_reversal", version="1", description="前日ピボットの R1/S1 で逆張り、P で利確")
class PivotReversal(Strategy):
    name = "pivot_reversal"

    @dataclass
    class Params:
        timeframe: int = 15
        min_range_pips: float = 20.0

    def prepare(self, bars, rng):
        ph, pl, pc = previous_day_hlc(bars.t, bars.high, bars.low, bars.close)
        self.P = (ph + pl + pc) / 3
        self.R1, self.S1 = 2 * self.P - pl, 2 * self.P - ph
        self.R2, self.S2 = self.P + (ph - pl), self.P - (ph - pl)
        self.rng_ = ph - pl
        self.c = bars.close
        self.day = trading_day(bars.t)
        self.done: set = set()

    def on_bar(self, i, ctx):
        c, P = self.c[i], self.P[i]
        if ctx.side != 0 or not np.isfinite(P) or self.rng_[i] < self.p.min_range_pips * ctx.pip:
            return
        d = int(self.day[i])
        if c >= self.R1[i] and (d, -1) not in self.done and c < self.R2[i]:
            ctx.sell(sl=self.R2[i] - c, tp=c - P)
            self.done.add((d, -1))
        elif c <= self.S1[i] and (d, 1) not in self.done and c > self.S2[i]:
            ctx.buy(sl=c - self.S2[i], tp=P - c)
            self.done.add((d, 1))
