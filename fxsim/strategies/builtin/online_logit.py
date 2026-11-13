"""S024 online_logit: オンライン学習のロジスティック回帰で、horizon 本後の向きを予測する(1 時間足)。

特徴量: 1・4・24 本のリターン(ATR で割る)、RSI、24 本 EMA からの乖離(ATR で割る)、時刻(sin / cos)。
足 i の判断に使うモデルは、答え(足 j から horizon 本後の向き)が足 i の時点でもう分かっている例(j ≤ i − horizon)だけで
確率的勾配法により少しずつ更新する(先読みしない)。予測した確率が 0.5 ± margin を超えたら入り、horizon 本持つ。
機械学習を流行りの道具として使ったときに何が起きるかを見る基準。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, ema, rsi


@STRATEGIES.register("online_logit", version="1", description="オンライン学習のロジスティック回帰で数本先の向きを予測")
class OnlineLogit(Strategy):
    name = "online_logit"

    @dataclass
    class Params:
        timeframe: int = 60
        horizon: int = 6
        lr: float = 0.01
        l2: float = 1e-4
        margin: float = 0.05
        warm: int = 2000
        atr_n: int = 24
        sl_atr: float = 2.5

    def prepare(self, bars, rng):
        p = self.p
        c = bars.close
        a = atr(bars.high, bars.low, c, p.atr_n)
        a = np.where(a > 0, a, np.nan)

        def ret(k):
            return np.r_[np.full(k, np.nan), c[k:] - c[:-k]] / a

        hour = ((bars.t // 3_600_000_000_000) % 24) / 24 * 2 * np.pi
        X = np.column_stack([np.ones(len(c)), ret(1), ret(4), ret(24), (rsi(c, 14) - 50) / 25,
                             (c - ema(c, 24)) / a, np.sin(hour), np.cos(hour)])
        X = np.clip(np.nan_to_num(X), -5, 5)
        h = p.horizon
        y = np.r_[(c[h:] > c[:-h]).astype(float), np.full(h, np.nan)]   # 足 j の答え(足 j + h で分かる)
        n, d = X.shape
        w = np.zeros(d)
        prob = np.full(n, 0.5)
        for i in range(n):
            j = i - h                                   # 足 i の終値で答えが分かった例
            if j >= 0:
                pj = 1 / (1 + np.exp(-X[j] @ w))
                w += p.lr * ((y[j] - pj) * X[j] - p.l2 * w)
            prob[i] = 1 / (1 + np.exp(-X[i] @ w))
        self.prob = prob
        self.warmup = p.warm
        self.atr = a

    def on_bar(self, i, ctx):
        p = self.p
        if ctx.side != 0:
            if i - ctx.position.entry_idx >= p.horizon:
                ctx.close()
            return
        q, a = self.prob[i], self.atr[i]
        if not np.isfinite(a):
            return
        if q > 0.5 + p.margin:
            ctx.buy(sl=p.sl_atr * a)
        elif q < 0.5 - p.margin:
            ctx.sell(sl=p.sl_atr * a)
