"""S015 parabolic_sar: パラボリック SAR のドテン売買(4 時間足)。

SAR の向きが変わったら反対へ入り直す(常に持つ)。損切りは毎本 SAR の位置へ動かす(トレーリングストップ)。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import parabolic_sar


@STRATEGIES.register("parabolic_sar", version="1", description="パラボリック SAR でドテン、SAR を損切りとして追う")
class ParabolicSar(Strategy):
    name = "parabolic_sar"

    @dataclass
    class Params:
        timeframe: int = 240
        step: float = 0.02
        max_step: float = 0.2

    def prepare(self, bars, rng):
        self.warmup = 5
        self.sar, self.trend = parabolic_sar(bars.high, bars.low, self.p.step, self.p.max_step)
        self.c = bars.close

    def on_bar(self, i, ctx):
        tr, s, c = int(self.trend[i]), self.sar[i], self.c[i]
        if tr == 0 or not np.isfinite(s):
            return
        if ctx.side == tr:
            ctx.set_sl(s)                                   # SAR を追いかける
            return
        dist = abs(c - s)
        if dist <= 0:
            return
        if tr > 0:
            ctx.buy(sl=dist)
        else:
            ctx.sell(sl=dist)
