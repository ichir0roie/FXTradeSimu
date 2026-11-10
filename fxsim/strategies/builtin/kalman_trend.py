"""S025 kalman_trend: カルマンフィルタで価格の「水準」と「傾き」を推定し、傾きが十分はっきりしている向きへ(4 時間足)。

状態 = (水準, 傾き) の局所線形トレンドモデル。観測ノイズ・状態ノイズは、直近のリターンの分散から決める。
傾き / その標準偏差 が ±threshold を超えたらその向きへ、0 をまたいだら手仕舞い。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr


@STRATEGIES.register("kalman_trend", version="1",
                     description="カルマンフィルタの傾きの向きへ(はっきりしているときだけ)")
class KalmanTrend(Strategy):
    name = "kalman_trend"

    @dataclass
    class Params:
        timeframe: int = 240
        q_level: float = 0.1            # 状態ノイズ(水準)/ 1 本のリターンの分散
        q_slope: float = 0.001          # 状態ノイズ(傾き)/ 同
        r_obs: float = 1.0              # 観測ノイズ / 同
        threshold: float = 2.0
        atr_n: int = 14
        sl_atr: float = 3.0
        var_n: int = 200

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = p.var_n
        y = np.log(bars.close)
        r = np.diff(y, prepend=y[0])
        n = len(y)
        x = np.array([y[0], 0.0])
        P = np.eye(2) * 1e-4
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([1.0, 0.0])
        self.t = np.zeros(n)
        v = 1e-6
        for i in range(n):
            if i >= 2:
                lo = max(1, i - p.var_n)
                v = max(float(np.mean(r[lo:i + 1] ** 2)), 1e-12)     # 足 i までのリターンの二乗平均
            Q = np.diag([p.q_level * v, p.q_slope * v])
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H + p.r_obs * v
            K = P @ H / S
            x = x + K * (y[i] - H @ x)
            P = P - np.outer(K, H) @ P
            self.t[i] = x[1] / np.sqrt(max(P[1, 1], 1e-30))
        self.atr = atr(bars.high, bars.low, bars.close, p.atr_n)

    def on_bar(self, i, ctx):
        t, a = self.t[i], self.atr[i]
        if not np.isfinite(a):
            return
        if ctx.side != 0 and ctx.side * t <= 0:
            ctx.close()
            return
        if ctx.side == 0:
            if t > self.p.threshold:
                ctx.buy(sl=self.p.sl_atr * a)
            elif t < -self.p.threshold:
                ctx.sell(sl=self.p.sl_atr * a)
