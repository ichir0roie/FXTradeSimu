"""S900 direction_null: 別の戦略の「方向だけでたらめ版」(検証の帰無仮説)。

中の戦略(inner)を同じように動かし、新規に建てるたびに買い/売りを乱数で決め直す。
売買のタイミング・保有期間・損切り/利確の幅・取引回数・コストは中の戦略と同じなので、
同じ相場パターンで比べた損益の差が、中の戦略の「方向を当てる力」になる。
中の戦略からは、自分が建てたつもりの向き(仮想の向き)が見える。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import STRATEGIES
from ..base import Strategy


class _FlippedPosition:
    """中の戦略に見せる建玉。向きだけ仮想の向きにする。"""

    def __init__(self, pos, flip: int):
        self._pos = pos
        self.side = pos.side * flip

    def __getattr__(self, name):
        return getattr(self._pos, name)


class _FlipContext:
    def __init__(self, owner: DirectionNull, ctx):
        self._o = owner
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    @property
    def side(self) -> int:
        return self._ctx.side * self._o.flip

    @property
    def position(self):
        pos = self._ctx.position
        return None if pos is None else _FlippedPosition(pos, self._o.flip)

    def _enter(self, side: int, **kw) -> None:
        o = self._o
        if not self._ctx.can_enter:
            if self.side == -side:
                self._ctx.close()
            return
        o.flip = 1 if o.u[o.i] < 0.5 else -1
        (self._ctx.buy if side * o.flip > 0 else self._ctx.sell)(**kw)

    def buy(self, **kw) -> None:
        self._enter(1, **kw)

    def sell(self, **kw) -> None:
        self._enter(-1, **kw)


@STRATEGIES.register("direction_null", version="1",
                     description="別の戦略の売買タイミングのまま、方向だけ乱数にしたもの(検証の帰無仮説)")
class DirectionNull(Strategy):
    name = "direction_null"

    @dataclass
    class Params:
        inner: str = "ma_cross"
        inner_params: dict = field(default_factory=dict)
        timeframe: int = 60

    def __init__(self, **params):
        super().__init__(**params)
        inner_params = {**self.p.inner_params, "timeframe": self.p.timeframe}
        self.inner = STRATEGIES.create(self.p.inner, inner_params)
        self.flip = 1
        self.i = 0

    def prepare(self, bars, rng):
        self.u = rng.random(len(bars))          # 先に全部引く(先読み検査で同じ値になるように)
        self.inner.prepare(bars, rng)
        self.warmup = getattr(self.inner, "warmup", 0)

    def on_bar(self, i, ctx):
        self.i = i
        if ctx.side == 0:
            self.flip = 1
        self.inner.on_bar(i, _FlipContext(self, ctx))
