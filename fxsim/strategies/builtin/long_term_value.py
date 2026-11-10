"""S021 long_term_value: 長期の平均への回帰(日足)。購買力平価のような「適正な水準」の代わりに、
5 年(1260 日)の対数価格の平均からのずれ(z 値)を使い、±entry_z を超えたら逆張りで持ち、z が 0 に戻るまで持つ。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import zscore


@STRATEGIES.register("long_term_value", version="1", description="5 年平均からの大きなずれに逆張り、平均に戻るまで持つ")
class LongTermValue(Strategy):
    name = "long_term_value"

    @dataclass
    class Params:
        timeframe: int = 1440
        n: int = 1260
        entry_z: float = 1.5
        max_hold: int = 750

    def prepare(self, bars, rng):
        self.warmup = self.p.n
        self.z = zscore(np.log(bars.close), self.p.n)

    def on_bar(self, i, ctx):
        z = self.z[i]
        if not np.isfinite(z):
            return
        if ctx.side != 0:
            held = i - ctx.position.entry_idx
            if ctx.side * z >= 0 or held >= self.p.max_hold:
                ctx.close()
            return
        if z > self.p.entry_z:
            ctx.sell()
        elif z < -self.p.entry_z:
            ctx.buy()
