"""S031 nanpin_martingale: ナンピン・マーチンゲール(1 時間足)。日本の個人の FX でよく知られた、そしてよく破綻する型。

coin(コイン投げ)か trend の向きで 1 単位入り、step_pips 逆に動くごとに、直前の量の mult 倍を同じ向きに足す
(平均取得単価を下げる)。
平均取得単価から tp_pips 戻ったら全部利確。足すのは max_adds 回まで。それを超えても戻らなければ、
平均から stop_pips(0 なら損切りなし = ロスカットまで持つ)で損切り。
ほとんどの回で小さく勝ち、まれに口座の大半を失う。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import ema


@STRATEGIES.register("nanpin_martingale", version="1",
                     description="逆行するたびに倍の量でナンピンし、平均から少し戻ったら全部利確")
class NanpinMartingale(Strategy):
    name = "nanpin_martingale"

    @dataclass
    class Params:
        timeframe: int = 60
        direction: str = "coin"        # coin | trend
        step_pips: float = 30.0
        mult: float = 2.0
        max_adds: int = 5
        tp_pips: float = 10.0
        stop_pips: float = 0.0
        trend_n: int = 100

    def prepare(self, bars, rng):
        self.flip = rng.random(len(bars)) < 0.5
        self.trend = np.sign(bars.close - ema(bars.close, self.p.trend_n))
        self.warmup = self.p.trend_n if self.p.direction == "trend" else 0
        self.c = bars.close
        self.adds = 0
        self.last_price = np.nan
        self.last_units = 0.0

    def on_bar(self, i, ctx):
        p, c, pip = self.p, self.c[i], ctx.pip
        side = ctx.side
        if side == 0:
            self.adds = 0
            d = (1 if self.flip[i] else -1) if p.direction == "coin" else int(self.trend[i])
            if d == 0:
                return
            u = ctx.base_units
            stop = p.stop_pips * pip if p.stop_pips > 0 and p.max_adds == 0 else None
            (ctx.buy if d > 0 else ctx.sell)(tp=p.tp_pips * pip, sl=stop, units=u)
            self.last_price, self.last_units = c, u
            return
        if self.adds < p.max_adds and (self.last_price - c) * side >= p.step_pips * pip:
            u = self.last_units * p.mult
            self.adds += 1
            last = self.adds == p.max_adds
            stop = p.stop_pips * pip if (last and p.stop_pips > 0) else None
            ctx.add(u, tp=p.tp_pips * pip, sl=stop, sl_from_average=True)
            self.last_price, self.last_units = c, u
