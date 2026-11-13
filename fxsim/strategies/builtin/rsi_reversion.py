"""S008 rsi_reversion: RSI が 30 を割ったら買い、70 を超えたら売り(オシレーターの逆張りの定番、1 時間足)。

RSI が 50 に戻ったら手仕舞い。損切りは ATR の倍数、max_hold 本で時間切れ。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, rsi


@STRATEGIES.register("rsi_reversion", version="1", description="RSI 30/70 の逆張り、50 で手仕舞い")
class RsiReversion(Strategy):
    name = "rsi_reversion"

    @dataclass
    class Params:
        timeframe: int = 60
        n: int = 14
        low: float = 30.0
        high: float = 70.0
        exit_level: float = 50.0
        atr_n: int = 24
        sl_atr: float = 2.5
        max_hold: int = 48

    def prepare(self, bars, rng):
        self.warmup = max(self.p.n, self.p.atr_n) + 1
        self.r = rsi(bars.close, self.p.n)
        self.atr = atr(bars.high, bars.low, bars.close, self.p.atr_n)

    def on_bar(self, i, ctx):
        r, a, p = self.r[i], self.atr[i], self.p
        if not (np.isfinite(r) and np.isfinite(a)):
            return
        if ctx.side != 0:
            held = i - ctx.position.entry_idx
            if (ctx.side > 0 and r >= p.exit_level) or (ctx.side < 0 and r <= p.exit_level) or held >= p.max_hold:
                ctx.close()
            return
        if r < p.low:
            ctx.buy(sl=p.sl_atr * a)
        elif r > p.high:
            ctx.sell(sl=p.sl_atr * a)
