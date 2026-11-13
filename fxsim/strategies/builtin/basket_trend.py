"""S035 basket_trend: 複数ペアのトレンドフォロー(CTA の型を為替 7 ペアで)。

ペアごとに、lookback 本の log リターンが正なら買い、負なら売る(順位ではなく、ペアごとの時系列の向き)。
建玉の金額は、直近 vol_window 本の実現ボラの逆数に比例させ、全ペアのリスクをそろえる
(target_vol は口座全体の年率の目安)。
hold_days 本ごとに向きを見直す。1 ペアのトレンドフォロー(tsmom・turtle)との違いは分散だけ。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..basket import BasketStrategy, closes, every, lagged_log_return, rolling_vol


@STRATEGIES.register("basket_trend", version="1", description="7 ペアそれぞれの向きに乗る(ボラで量をそろえる)")
class BasketTrend(BasketStrategy):
    name = "basket_trend"

    @dataclass
    class Params:
        timeframe: int = 1440
        lookback: int = 126
        hold_days: int = 5
        vol_window: int = 63
        target_vol: float = 0.10      # 口座全体の年率ボラの目安(ペアどうしの相関は見ない)
        max_gross: float = 3.0        # 建玉の総額 / 初期資金 の上限

    def prepare(self, bars_by_pair, rng):
        self.pairs, c = closes(bars_by_pair)
        self.c = c
        mom = lagged_log_return(c, self.p.lookback)
        self.side = np.where(np.isfinite(mom), np.sign(mom), 0).astype(int)
        self.vol = rolling_vol(c, self.p.vol_window)
        first = max(self.p.lookback, self.p.vol_window)
        self.warmup = first
        self.rebalance = every(len(c), first, self.p.hold_days)

    def on_bar(self, i, ctx):
        if not self.rebalance[i]:
            return
        n = len(self.pairs)
        v = self.vol[i]
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            return
        # ペアごとの金額 = 初期資金 x (target_vol / √n) / ボラ。合計が max_gross を超えたら縮める
        notional = ctx.initial_balance * (self.p.target_vol / np.sqrt(n)) / v
        total = notional.sum() / ctx.initial_balance
        if total > self.p.max_gross:
            notional *= self.p.max_gross / total
        for j, pair in enumerate(self.pairs):
            units = float(np.floor(notional[j] / self.c[i, j] / 1000.0) * 1000.0)
            ctx.target(pair, int(self.side[i, j]) if units >= 1000 else 0, units)
