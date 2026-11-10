"""S003 bollinger_reversion: ボリンジャーバンドの外側で逆張りし、中心線に戻ったら出る。

損切りは ATR 基準、max_hold 本で時間切れ。群衆の逆突き(順張り相場)やトレンド局面で焼かれる型。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, rolling_std, sma


@STRATEGIES.register("bollinger_reversion", version="1", description="±k σ の外で逆張り、中心線で決済(平均回帰)")
class BollingerReversion(Strategy):
    name = "bollinger_reversion"

    @dataclass
    class Params:
        timeframe: int = 60
        n: int = 24
        k: float = 2.0
        atr_n: int = 24
        sl_atr: float = 2.5
        max_hold: int = 48

    def prepare(self, bars, rng):
        self.warmup = self.p.n
        c = bars.close
        m = sma(c, self.p.n)
        sd = rolling_std(c, self.p.n)
        self.z = (c - m) / np.where(sd > 0, sd, np.nan)
        self.atr = atr(bars.high, bars.low, c, self.p.atr_n)

    def on_bar(self, i, ctx):
        z, a = self.z[i], self.atr[i]
        if not (np.isfinite(z) and np.isfinite(a)):
            return
        side = ctx.side
        if side != 0:
            held = i - ctx.position.entry_idx
            if (side > 0 and z >= 0) or (side < 0 and z <= 0) or held >= self.p.max_hold:
                ctx.close()
            return
        if z < -self.p.k:
            ctx.buy(sl=self.p.sl_atr * a)
        elif z > self.p.k:
            ctx.sell(sl=self.p.sl_atr * a)
