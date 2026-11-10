"""S023 regime_adaptive: 相場の性格を測って、順張りと逆張りを切り替える(1 時間足)。

直近 window 本の分散比 VR(k) = Var(k 本のリターン) / (k x Var(1 本のリターン)) を測る。
VR < 1 − band(戻りやすい)なら z 値の逆張り、VR > 1 + band(走りやすい)なら直近 k 本の向きへ順張り、間なら何もしない。
「どの手法が効くかは時期による」への、よく使われる答え方。悪意ある相場はそれをさらに裏返してくる。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, zscore


@STRATEGIES.register("regime_adaptive", version="1", description="分散比で順張り/逆張りを切り替える")
class RegimeAdaptive(Strategy):
    name = "regime_adaptive"

    @dataclass
    class Params:
        timeframe: int = 60
        k: int = 8
        window: int = 500
        band: float = 0.1
        z_n: int = 24
        z_entry: float = 2.0
        hold: int = 12
        atr_n: int = 24
        sl_atr: float = 2.5

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = p.window + p.k
        lc = np.log(bars.close)
        r1 = np.diff(lc, prepend=lc[0])
        rk = np.r_[np.zeros(p.k), lc[p.k:] - lc[:-p.k]]
        v1 = pd.Series(r1).rolling(p.window).var().to_numpy(copy=True)
        vk = pd.Series(rk).rolling(p.window).var().to_numpy(copy=True)
        self.vr = vk / (p.k * np.where(v1 > 0, v1, np.nan))
        self.mom = np.sign(rk)
        self.z = zscore(bars.close, p.z_n)
        self.atr = atr(bars.high, bars.low, bars.close, p.atr_n)

    def on_bar(self, i, ctx):
        p = self.p
        if ctx.side != 0:
            if i - ctx.position.entry_idx >= p.hold:
                ctx.close()
            return
        vr, z, a = self.vr[i], self.z[i], self.atr[i]
        if not np.all(np.isfinite([vr, z, a])):
            return
        if vr < 1 - p.band and abs(z) > p.z_entry:
            (ctx.sell if z > 0 else ctx.buy)(sl=p.sl_atr * a)
        elif vr > 1 + p.band and self.mom[i] != 0:
            (ctx.buy if self.mom[i] > 0 else ctx.sell)(sl=p.sl_atr * a)
