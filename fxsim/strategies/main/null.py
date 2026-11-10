"""S900 の本命版 main_direction_null: 本命の戦略の「方向だけでたらめ版」(検証の帰無仮説)。

中の戦略(MainStrategy)をそのまま MT5 のテスターで動かし、新規の向きだけを乱数で決める。手仕舞いの時期は元の向きの
信号で決めるので、売買の時期・保有期間・量・損切りの幅・費用の門・回転の上限は中の戦略と同じになる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import STRATEGIES
from .base import MainStrategy


@STRATEGIES.register("main_direction_null", version="1",
                     description="本命の戦略の売買の時期のまま、新規の向きだけ乱数にしたもの(検証の帰無仮説)")
class MainDirectionNull(MainStrategy):
    name = "main_direction_null"

    @dataclass
    class Params:
        inner: str = "main_trend_d1"
        inner_params: dict = field(default_factory=dict)
        timeframe: int = 1440

    def __init__(self, **params):
        super().__init__(**params)
        self.inner = STRATEGIES.create(self.p.inner, {**self.p.inner_params, "timeframe": self.p.timeframe})
        if not isinstance(self.inner, MainStrategy):
            raise TypeError(f"{self.p.inner} is not a main strategy")
        self.magic = self.inner.magic

    def prepare(self, bars, rng):
        u = rng.random(len(bars))          # 先に全部引く(先読み検査で同じ値になるように)
        self.inner.prepare(bars, rng)
        self.inner.null_u = u
        self.warmup = getattr(self.inner, "warmup", 0)

    def on_bar(self, i, ctx):
        self.inner.on_bar(i, ctx)
