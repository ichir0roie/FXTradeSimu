"""S006 tsmom: 時系列モメンタム(日足)。

過去 lookbacks 日それぞれのリターンの符号を多数決し、プラスなら買い、マイナスなら売りで持ち続ける。
向きが変わったときだけ売買する(ドテン)。保険の損切りは ATR の倍数(0 なら無し)。
為替の時系列モメンタムは昔から知られた定番で、パラメータは結果を見て調整していない(既定 20 / 60 / 120 日)。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr


@STRATEGIES.register("tsmom", version="1",
                     description="過去 20/60/120 日のリターンの符号の多数決で持ち続ける(時系列モメンタム)")
class TimeSeriesMomentum(Strategy):
    name = "tsmom"

    @dataclass
    class Params:
        timeframe: int = 1440
        lookbacks: tuple = (20, 60, 120)
        atr_n: int = 20
        sl_atr: float = 0.0

    def prepare(self, bars, rng):
        c = bars.close
        n = max(self.p.lookbacks)
        self.warmup = n
        votes = np.zeros(len(c))
        for k in self.p.lookbacks:
            r = np.full(len(c), np.nan)
            r[k:] = c[k:] / c[:-k] - 1          # 足 i の終値までの k 本のリターン(過去だけ)
            votes += np.sign(np.nan_to_num(r))
        self.want = np.sign(votes)
        self.want[:n] = 0
        self.atr = atr(bars.high, bars.low, c, self.p.atr_n)

    def on_bar(self, i, ctx):
        want = int(self.want[i])
        if want == 0 or ctx.side == want:
            return
        sl = self.p.sl_atr * self.atr[i] if self.p.sl_atr > 0 and np.isfinite(self.atr[i]) else None
        if want > 0:
            ctx.buy(sl=sl)
        else:
            ctx.sell(sl=sl)
