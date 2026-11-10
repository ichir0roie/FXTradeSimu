"""S901 basket_direction_null: 複数ペアの戦略の「方向だけでたらめ版」(検証の帰無仮説)。

中の戦略(inner)を同じように動かし、あるペアを新しく建てる・向きを変えるたびに、そのペアの買い/売りを乱数で決め直す。
入れ替えのタイミング・ペアの数・量・費用は中の戦略と同じなので、同じ相場パターンで比べた損益の差が、
中の戦略の「どのペアをどちら向きに持つか」を当てる力になる。中の戦略からは、自分が建てたつもりの向きが見える。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import STRATEGIES
from ..basket import BasketStrategy


class _FlipContext:
    def __init__(self, owner: BasketDirectionNull, ctx):
        self._o = owner
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def side(self, pair: str) -> int:
        return self._ctx.side(pair) * self._o.flip.get(pair, 1)

    def target(self, pair: str, side: int, units=None) -> None:
        o = self._o
        side = int(np.sign(side))
        if side == self.side(pair):
            return
        if side != 0:
            o.flip[pair] = 1 if o.rng.random() < 0.5 else -1
        self._ctx.target(pair, side * o.flip.get(pair, 1), units)

    def close_all(self) -> None:
        for p in self._ctx.pairs:
            self.target(p, 0)


@STRATEGIES.register("basket_direction_null", version="1",
                     description="複数ペアの戦略の入れ替えのまま、ペアごとの向きだけ乱数にしたもの(検証の帰無仮説)")
class BasketDirectionNull(BasketStrategy):
    name = "basket_direction_null"

    @dataclass
    class Params:
        inner: str = "xs_carry"
        inner_params: dict = field(default_factory=dict)
        timeframe: int = 1440

    def __init__(self, **params):
        super().__init__(**params)
        self.inner = STRATEGIES.create(self.p.inner, {**self.p.inner_params, "timeframe": self.p.timeframe})
        self.flip: dict[str, int] = {}

    def prepare(self, bars_by_pair, rng):
        # 乱数は向きを決め直すたびに引く。中の戦略の判断は因果的なので、先頭を切った相場でも同じ順に引かれる
        self.rng = np.random.default_rng(rng.integers(0, 2**63))
        self.inner.prepare(bars_by_pair, rng)
        self.warmup = getattr(self.inner, "warmup", 0)

    def on_bar(self, i, ctx):
        self.inner.on_bar(i, _FlipContext(self, ctx))
