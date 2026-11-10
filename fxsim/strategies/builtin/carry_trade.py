"""S020 carry_trade: キャリートレード(日足)。金利の高い通貨を買って持ち、スワップを受け取る。

ドル円なら、米国の政策金利が日本より min_diff 以上高ければ買い、低ければ売り、差が小さければ持たない。
trend_n > 0 なら、終値が trend_n 日の移動平均と同じ側にあるときだけ持つ(キャリーの巻き戻しを避ける、よくある工夫)。
金利の表は data/real/policy_rates_usd_jpy.csv(手入力の概算)。合成の相場の日付にもそのまま当てはめる。
"""
from dataclasses import dataclass

import numpy as np

from ...data import load_rates
from .. import STRATEGIES
from ..base import Strategy
from ..indicators import sma


@STRATEGIES.register("carry_trade", version="1",
                     description="金利の高い側を持ち続けてスワップを取る(移動平均の向きで絞れる)")
class CarryTrade(Strategy):
    name = "carry_trade"

    @dataclass
    class Params:
        timeframe: int = 1440
        min_diff: float = 0.01          # 年率(0.01 = 1%)
        trend_n: int = 0
        rates_path: str = "data/real/policy_rates_usd_jpy.csv"

    def prepare(self, bars, rng):
        self.warmup = self.p.trend_n
        diff = load_rates(self.p.rates_path).differential(bars.t)
        want = np.where(diff >= self.p.min_diff, 1, np.where(diff <= -self.p.min_diff, -1, 0))
        if self.p.trend_n > 0:
            m = sma(bars.close, self.p.trend_n)
            agree = np.sign(bars.close - m) == want
            want = np.where(agree & np.isfinite(m), want, 0)
        self.want = want

    def on_bar(self, i, ctx):
        w = int(self.want[i])
        if w == ctx.side:
            return
        if w == 0:
            ctx.close()
        elif w > 0:
            ctx.buy()
        else:
            ctx.sell()
