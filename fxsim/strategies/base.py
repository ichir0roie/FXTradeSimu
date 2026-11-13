"""戦略の基底クラス。

書き方:
    1. Params(dataclass)にパラメータを書く。timeframe(分)は共通で持つ。
    2. prepare(bars, rng) で指標を numpy でまとめて計算しておく(因果的な計算だけ。shift(-1) などは禁止)。
    3. on_bar(i, ctx) で足 i の終値時点の判断をする。ctx.buy / sell / close / set_sl。
       注文は次の足の始値で約定する。
    4. ``@STRATEGIES.register("名前")`` で登録し、builtin/__init__.py で import する。
複数ペアの戦略は strategies/basket.py の BasketStrategy を継承する(multi = True)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import ClassVar

import numpy as np

from ..backtest.engine import StrategyContext
from ..data import Bars
from ..registry import params_hash


class Strategy:
    name: ClassVar[str] = "strategy"
    version: ClassVar[str] = "1"
    multi: ClassVar[bool] = False       # True なら複数ペアの戦略(strategies/basket.py、相場は basket_*)
    # 戦略の区分: basic(基本検証。strategies/builtin/ の手法の一覧。E011 で全部不合格)/
    # main(本命。strategies/main/。MT5 に移す前提で、共通の対策(strategies/guards.py)を入れて作る)
    tier: ClassVar[str] = "basic"
    # 回すエンジン: classic(backtest/engine.py、複数ペアは portfolio.py)/ mt5(backtest/mt5 の Mt5Tester)
    engine: ClassVar[str] = "classic"

    @dataclass
    class Params:
        timeframe: int = 60

    warmup: int = 0

    def __init__(self, **params):
        known = {f.name for f in fields(self.Params)}
        unknown = set(params) - known
        if unknown:
            raise KeyError(f"{self.name}: unknown params {sorted(unknown)} (known: {sorted(known)})")
        self.p = self.Params(**params)

    @property
    def timeframe(self) -> int:
        return int(self.p.timeframe)

    def prepare(self, bars: Bars, rng: np.random.Generator) -> None:
        pass

    def on_bar(self, i: int, ctx: StrategyContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> dict:
        params = asdict(self.p)
        return {"name": self.name, "version": self.version, "params": params, "params_hash": params_hash(params)}
