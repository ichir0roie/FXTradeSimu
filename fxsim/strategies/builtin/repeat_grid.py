"""S037 repeat_grid: リピート系の自動売買(トラリピ・ループイフダンの型)。

買いのリピート(side=long)なら、起点 anchor から step_pct % 刻みに n_levels 段の買いの値段を並べる
(anchor, anchor x (1 − step), anchor x (1 − 2 step), ...)。
終値がある段の値段以下になったらその段を 1 段分(units_per_level)買い、
その段の値段 x (1 + step) 以上になったら 1 段分を利確する。利確した段は、また下がれば買い直す(繰り返し)。
損切りは置かない(国内のリピート系の典型的な使い方)。証拠金が足りなくなればブローカーのロスカットで終わる。
売りのリピート(side=short)は上下を反転したもの。

近似: 本物は指値で約定するが、ここは判断の足(既定 1 時間足)の終値で判定し、次の足の始値の成行で約定する
(足の中の細かい往復は取りこぼす。指値と違ってスプレッドと滑りを払う)。
建玉はペアで 1 つ(平均取得単価)にまとめ、段ごとの利確は一部決済(reduce)で表す。損益の合計は段ごとに持つのと同じ。
レンジの付け替え: 全段が空のまま recenter_days 日たち、値段がレンジの外(買いなら上)にあるときは、
今の終値を新しい起点にする
(放っておくと上に抜けた相場で何もしなくなる。利用者が設定し直すのを真似る。0 なら付け替えない)。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy


@STRATEGIES.register("repeat_grid", version="1",
                     description="値幅ごとに買い(売り)を並べ、1 段ずつ利確して繰り返す(リピート系、損切りなし)")
class RepeatGrid(Strategy):
    name = "repeat_grid"

    @dataclass
    class Params:
        timeframe: int = 60
        side: str = "long"            # long(買いのリピート)| short(売りのリピート)
        step_pct: float = 0.5         # 段の間隔(%)。利確の幅も同じ
        n_levels: int = 30
        units_per_level: float = 1000.0
        recenter_days: float = 60.0

    def __init__(self, **params):
        super().__init__(**params)
        if self.p.side not in ("long", "short"):
            raise ValueError("side must be long or short")

    def prepare(self, bars, rng):
        self.close = bars.close
        self.t = bars.t
        self.sign = 1 if self.p.side == "long" else -1
        self.anchor = None
        self.filled = np.zeros(self.p.n_levels, dtype=bool)
        self.flat_since = None

    def _set_anchor(self, price: float) -> None:
        s = self.p.step_pct / 100
        k = np.arange(self.p.n_levels)
        self.anchor = price
        self.levels = price * (1 - self.sign * s * k)             # 買いなら下へ、売りなら上へ並ぶ
        self.targets = self.levels * (1 + self.sign * s)          # 利確の値段

    def on_bar(self, i, ctx):
        c = float(self.close[i])
        if self.anchor is None:
            self._set_anchor(c)
        sg = self.sign
        if not self.filled.any():
            if self.flat_since is None:
                self.flat_since = self.t[i]
            elif (self.p.recenter_days > 0 and (self.t[i] - self.flat_since) / 86_400e9 >= self.p.recenter_days
                  and sg * (c - self.levels[0]) > 0):
                self._set_anchor(c)
                self.flat_since = self.t[i]
        else:
            self.flat_since = None
        buy = ~self.filled & (sg * (c - self.levels) <= 0)
        take = self.filled & (sg * (c - self.targets) >= 0)
        self.filled = (self.filled | buy) & ~take
        net = int(buy.sum()) - int(take.sum())
        if net == 0:
            return
        units = abs(net) * self.p.units_per_level
        if net > 0:
            if ctx.side == 0:
                (ctx.buy if sg > 0 else ctx.sell)(units=units)
            else:
                ctx.add(units)
        else:
            if not self.filled.any():
                ctx.close()
            else:
                ctx.reduce(units)
