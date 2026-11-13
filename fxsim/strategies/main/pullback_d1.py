"""S048 main_pullback_d1: 長いトレンドの向きに、深い押し目(戻り)で入り、数日で出る。

E048 のふるい(日足の 184 型、主要 10 ペアの実データ)で、発見(1996〜2013)と確認(2014〜2026)の両方でプラスだった
ただ 1 つの族。形は教科書どおり(Connors の RSI(2))で、結果を見て数字を選んでいない:
終値が 200 日の移動平均より上で RSI(2) が 10 を割ったら買い、RSI(2) が 50 を越えるか 10 本たったら出る。売りは鏡。
損切りの幅は ATR(20) の 3 倍(終値で判定)、見込みの値幅は ATR(20) x √(持つ予定の日数)。
"""
from dataclasses import dataclass, field

import numpy as np

from .. import STRATEGIES
from ..indicators import atr, rsi, sma
from .base import MainStrategy


@STRATEGIES.register("main_pullback_d1", version="1",
                     description="本命: 日足の押し目(200 日平均の向き + RSI(2) の行き過ぎ)に共通の対策(MT5 前提)")
class MainPullbackD1(MainStrategy):
    name = "main_pullback_d1"

    @dataclass
    class Params:
        timeframe: int = 1440
        guard: dict = field(default_factory=lambda: {"hold_days_est": 3.0, "max_trades_per_year": 40.0})
        slow: int = 200
        rsi_n: int = 2
        enter: float = 10.0          # 買いは RSI < enter、売りは RSI > 100 − enter
        exit: float = 50.0           # 買いは RSI > exit、売りは RSI < 100 − exit で出る
        max_bars: int = 10
        atr_n: int = 20
        stop_atr: float = 3.0
        trend: bool = True           # False なら移動平均の向きを見ない(ただの逆張り。対照)

    def signals(self, bars, rng):
        p = self.p
        c = bars.close
        n = len(c)
        ma = sma(c, p.slow)
        r = rsi(c, p.rsi_n)
        ok = np.isfinite(ma) & np.isfinite(r)
        up = (c > ma) if p.trend else np.ones(n, dtype=bool)
        dn = (c < ma) if p.trend else np.ones(n, dtype=bool)
        entry = np.where(ok & up & (r < p.enter), 1, np.where(ok & dn & (r > 100 - p.enter), -1, 0))
        target = np.zeros(n, dtype=int)
        cur, age = 0, 0
        for i in range(n):
            if cur != 0:
                age += 1
                if (r[i] > p.exit if cur > 0 else r[i] < 100 - p.exit) or age >= p.max_bars:
                    cur = 0
            if cur == 0 and entry[i] != 0:
                cur, age = int(entry[i]), 0
            target[i] = cur
        a = atr(bars.high, bars.low, c, p.atr_n)
        self.warmup = max(p.slow, p.atr_n)
        return target, p.stop_atr * a, a * np.sqrt(self.g.hold_days_est)
