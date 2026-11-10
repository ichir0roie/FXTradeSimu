"""S038 main_trend_d1: 本命の 1 本目。日足のトレンドに、共通の対策(guards)を全部入れたもの。

E012 で、日足の時系列モメンタム(B007)は費用の壁(1 取引 2.6 pip)を越える素の損益(+10 pip / 取引)を持っていたが、
スワップの上乗せ(年率 1%)で消えた。ここでは対策(回転の上限・費用とスワップの門・固定比率の量・終値の損切りと
遠くの非常用の逆指値・ドローダウンの止め)を入れて、MT5 の口座の決まりで回したときに何が残るかを見る。
信号は結果を見て調整していない定番の形:
終値が 200 日の移動平均より上で 60 日のリターンがプラスなら買い、両方逆なら売り。
損切りの幅は ATR(20) の 3 倍、見込みの値幅は ATR(20) x √(持つ予定の日数)。
"""
from dataclasses import dataclass, field

import numpy as np

from .. import STRATEGIES
from ..indicators import atr, sma
from .base import MainStrategy


@STRATEGIES.register("main_trend_d1", version="1",
                     description="本命: 日足のトレンド(200 日平均 + 60 日リターン)に共通の対策を入れた EA(MT5 前提)")
class MainTrendD1(MainStrategy):
    name = "main_trend_d1"

    @dataclass
    class Params:
        timeframe: int = 1440
        guard: dict = field(default_factory=dict)
        slow: int = 200
        mom: int = 60
        atr_n: int = 20
        stop_atr: float = 3.0

    def signals(self, bars, rng):
        c = bars.close
        n = len(c)
        ma = sma(c, self.p.slow)
        mom = np.full(n, np.nan)
        mom[self.p.mom:] = np.log(c[self.p.mom:] / c[:-self.p.mom])
        up = (c > ma) & (mom > 0)
        down = (c < ma) & (mom < 0)
        target = np.where(up, 1, np.where(down, -1, 0))
        target[~np.isfinite(ma) | ~np.isfinite(mom)] = 0
        a = atr(bars.high, bars.low, c, self.p.atr_n)
        self.warmup = max(self.p.slow, self.p.mom, self.p.atr_n)
        return target, self.p.stop_atr * a, a * np.sqrt(self.g.hold_days_est)
