"""S022 vol_target_trend: ボラティリティ・ターゲットのトレンドフォロー(日足)。CTA やリスクパリティで一般的な作り。

50 日と 200 日の EMA の向きへ持つ。量は「年率ボラ target_vol で動く量」= 有効証拠金 x target_vol / 実現ボラ(60 日)。
rebalance 日ごとに量を測り直す(持ち直す)。荒れたら小さく、静かなら大きく持つ。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import ema, rolling_std


@STRATEGIES.register("vol_target_trend", version="1", description="EMA 50/200 の向きへ、年率ボラを一定にする量で持つ")
class VolTargetTrend(Strategy):
    name = "vol_target_trend"

    @dataclass
    class Params:
        timeframe: int = 1440
        fast: int = 50
        slow: int = 200
        vol_n: int = 60
        target_vol: float = 0.10
        rebalance: int = 20
        max_leverage: float = 5.0

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = p.slow
        c = bars.close
        self.want = np.sign(ema(c, p.fast) - ema(c, p.slow))
        r = np.diff(np.log(c), prepend=np.log(c[0]))
        self.vol = rolling_std(r, p.vol_n) * np.sqrt(260)
        self.c = c
        self.last = -10 ** 9

    def on_bar(self, i, ctx):
        w, v = self.want[i], self.vol[i]
        if not (np.isfinite(w) and np.isfinite(v)) or v <= 0 or w == 0:
            return
        if ctx.side == w and i - self.last < self.p.rebalance:
            return
        lev = min(self.p.target_vol / v, self.p.max_leverage)
        units = ctx.equity * lev / self.c[i]
        if ctx.side == w:
            ctx.close()                      # 量を測り直すときは持ち直す(次の足で入り直す)
            return
        (ctx.buy if w > 0 else ctx.sell)(units=units)
        self.last = i
