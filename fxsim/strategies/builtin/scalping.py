"""S013 scalping: 数 pip を取りに行くスキャルピング(5 分足)。

ロンドン〜NY の時間(07〜17 UTC)に、20 本 EMA の向きへ、直近 6 本の高値・安値を抜けたら入る。
利確・損切りとも 5 pip。スプレッドと滑りが利幅に対して大きく、費用の重さが一番効く型。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import ema, highest, lowest


@STRATEGIES.register("scalping", version="1", description="EMA の向きに直近高安の抜けで入り、±5 pip で決着")
class Scalping(Strategy):
    name = "scalping"

    @dataclass
    class Params:
        timeframe: int = 5
        ema_n: int = 20
        lookback: int = 6
        tp_pips: float = 5.0
        sl_pips: float = 5.0
        start_hour: int = 7
        end_hour: int = 17

    def prepare(self, bars, rng):
        self.warmup = self.p.ema_n + 1
        c = bars.close
        self.e = ema(c, self.p.ema_n)
        self.hi = np.r_[np.nan, highest(bars.high, self.p.lookback)[:-1]]
        self.lo = np.r_[np.nan, lowest(bars.low, self.p.lookback)[:-1]]
        tc = bars.t + self.p.timeframe * 60_000_000_000
        h = (tc // 3_600_000_000_000) % 24
        self.ok = (h >= self.p.start_hour) & (h < self.p.end_hour)
        self.c = c

    def on_bar(self, i, ctx):
        if ctx.side != 0 or not self.ok[i] or i < 1:
            return
        c, e0, e1 = self.c[i], self.e[i - 1], self.e[i]
        if not (np.isfinite(e0) and np.isfinite(self.hi[i])):
            return
        pip = ctx.pip
        if e1 > e0 and c > self.hi[i]:
            ctx.buy(sl=self.p.sl_pips * pip, tp=self.p.tp_pips * pip)
        elif e1 < e0 and c < self.lo[i]:
            ctx.sell(sl=self.p.sl_pips * pip, tp=self.p.tp_pips * pip)
