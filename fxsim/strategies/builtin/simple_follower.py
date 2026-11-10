"""S004 simple_follower: 旧 samples/simpleFollower.py の移植。

直近 10 本の平均と、その前の 10 本の平均を比べ、下げ幅が drop を超えたら買う(押し目買い)。
平均が上向きに転じたら決済。平均は旧 iMa.simple と同じく (O+H+L+C)/4 の平均。
旧版は SL/TP を持つだけで判定していなかったので、ここでも既定では損切りなし。
旧版の閾値 1.0(価格)は 100 pip に当たり、ほぼ入らないので既定は 0.2(20 pip)にした。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import sma


@STRATEGIES.register("simple_follower", version="1", description="旧 simpleFollower: 平均の下げで買い、上向きで決済")
class SimpleFollower(Strategy):
    name = "simple_follower"

    @dataclass
    class Params:
        timeframe: int = 5
        window: int = 10
        drop: float = 0.2        # 価格単位
        sl: float = 0.0          # 価格単位。0 なら損切りなし

    def prepare(self, bars, rng):
        w = self.p.window
        self.warmup = 2 * w
        typical = (bars.open + bars.high + bars.low + bars.close) / 4
        after = sma(typical, w)
        before = np.r_[np.full(w, np.nan), after[:-w]]
        self.acc = after - before

    def on_bar(self, i, ctx):
        acc = self.acc[i]
        if not np.isfinite(acc):
            return
        if ctx.side == 0:
            if acc < -self.p.drop:
                ctx.buy(sl=self.p.sl or None)
        elif acc > 0:
            ctx.close()
