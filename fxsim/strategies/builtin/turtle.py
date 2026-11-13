"""S019 turtle: タートルズの売買ルール(システム 1、日足)。

- 入る: 20 日の高値(安値)を終値で抜けたら買い(売り)。ただし前回のシステム 1 の取引が勝ちなら見送る(本家の規則)
- 量: 1 ユニット = 有効証拠金の risk(1%)を、1N(20 日 ATR)動いたら失う量
- 積み増し: 0.5N 有利に動くごとに 1 ユニット足す(最大 4 ユニット)。損切りは最後に足した価格から 2N
- 出る: 10 日の安値(高値)を割ったら全部手仕舞い
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr, highest, lowest


@STRATEGIES.register("turtle", version="1", description="タートルズ: 20 日ブレイク、N で量を決め 0.5N ごとに積み増し")
class Turtle(Strategy):
    name = "turtle"

    @dataclass
    class Params:
        timeframe: int = 1440
        entry_n: int = 20
        exit_n: int = 10
        atr_n: int = 20
        risk: float = 0.01
        max_units: int = 4
        add_n: float = 0.5
        stop_n: float = 2.0
        skip_after_win: bool = True

    def prepare(self, bars, rng):
        p = self.p
        self.warmup = max(p.entry_n, p.atr_n) + 1
        self.hi = np.r_[np.nan, highest(bars.high, p.entry_n)[:-1]]
        self.lo = np.r_[np.nan, lowest(bars.low, p.entry_n)[:-1]]
        self.xhi = np.r_[np.nan, highest(bars.high, p.exit_n)[:-1]]
        self.xlo = np.r_[np.nan, lowest(bars.low, p.exit_n)[:-1]]
        self.N = atr(bars.high, bars.low, bars.close, p.atr_n)
        self.c = bars.close
        self.units_held = 0
        self.last_add = np.nan
        self.unit_size = 0.0
        self.n_trades_seen = 0
        self.last_win = False

    def on_bar(self, i, ctx):
        p, c, N = self.p, self.c[i], self.N[i]
        if not (np.isfinite(N) and np.isfinite(self.hi[i])) or N <= 0:
            return
        if len(ctx.trades) > self.n_trades_seen:              # 前回の取引の勝ち負け(見送りの規則)
            self.n_trades_seen = len(ctx.trades)
            self.last_win = ctx.trades[-1].pnl > 0
        side = ctx.side
        if side == 0:
            self.units_held = 0
        if side > 0 and c < self.xlo[i] or side < 0 and c > self.xhi[i]:
            ctx.close()
            return
        if side != 0:
            if self.units_held < p.max_units and (c - self.last_add) * side >= p.add_n * N:
                ctx.add(self.unit_size, sl=p.stop_n * N)
                self.units_held += 1
                self.last_add = c
            return
        go = 1 if c > self.hi[i] else -1 if c < self.lo[i] else 0
        if go == 0:
            return
        if p.skip_after_win and self.last_win:
            self.last_win = False                             # 1 回見送ったら次は入る
            return
        self.unit_size = max(1000.0, ctx.equity * p.risk / N)
        (ctx.buy if go > 0 else ctx.sell)(sl=p.stop_n * N, units=self.unit_size)
        self.units_held = 1
        self.last_add = c
