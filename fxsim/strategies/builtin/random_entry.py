"""S000 random_entry: 方向をランダムに決めて入る(帰無仮説)。

素の優位性はゼロのはずなので、成績はほぼ「コストと市場の悪意」だけを映す。
ほかの戦略の成績はこれと比べて読む。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr


@STRATEGIES.register("random_entry", version="1",
                     description="ランダムな方向に入り、ATR 基準の損切り/利確で出る(帰無仮説)")
class RandomEntry(Strategy):
    name = "random_entry"

    @dataclass
    class Params:
        timeframe: int = 60
        entry_prob: float = 0.05
        atr_n: int = 24
        sl_atr: float = 2.0
        tp_atr: float = 2.0

    warmup = 30

    def prepare(self, bars, rng):
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)
        # 行ごとに引く(長さが変わっても先頭の値が同じになり、先読み検査を通る)
        r = rng.random((len(bars), 2))
        self.u = r[:, 0]
        self.dir = r[:, 1] < 0.5

    def on_bar(self, i, ctx):
        if ctx.side != 0 or not np.isfinite(self.atr[i]) or self.u[i] >= self.p.entry_prob:
            return
        a = self.atr[i]
        if self.dir[i]:
            ctx.buy(sl=self.p.sl_atr * a, tp=self.p.tp_atr * a)
        else:
            ctx.sell(sl=self.p.sl_atr * a, tp=self.p.tp_atr * a)
