"""S001 ma_cross: 移動平均の交差に従うトレンドフォロー(ドテン)。保険の損切りは ATR の倍数。"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, ema


@STRATEGIES.register("ma_cross", version="1", description="EMA の短期/長期の交差でドテン(トレンドフォロー)")
class MACross(Strategy):
    name = "ma_cross"

    @dataclass
    class Params:
        timeframe: int = 60
        fast: int = 20
        slow: int = 100
        atr_n: int = 24
        sl_atr: float = 4.0      # 0 なら損切りなし

    def prepare(self, bars, rng):
        self.warmup = self.p.slow
        self.f = ema(bars.close, self.p.fast)
        self.s = ema(bars.close, self.p.slow)
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)

    def on_bar(self, i, ctx):
        f, s, a = self.f[i], self.s[i], self.atr[i]
        if not (np.isfinite(f) and np.isfinite(s) and np.isfinite(a)):
            return
        want = 1 if f > s else -1
        if ctx.side == want:
            return
        sl = self.p.sl_atr * a if self.p.sl_atr > 0 else None
        if want > 0:
            ctx.buy(sl=sl)
        else:
            ctx.sell(sl=sl)
