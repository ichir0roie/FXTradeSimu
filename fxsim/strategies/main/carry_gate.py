"""S043 main_carry_gate: 本命のスワップ狙い(main_carry_breakout)に、新規の関門を足したもの(E038)。

main_carry_breakout の動き(手仕舞い・口座の床・急変用の逆指値・量)はそのままで、新規だけを絞る。どれも 0 なら同じ。
    entry_vol_max   ATR(atr_n) / 直前 vol_window 本の ATR の中央値がこれを超えるペアには新規を出さない
                    (荒いときにキャリーを始めない。E036)
    counter_z       60 本の対数リターン / (20 本のボラ x √60) を建てる向きで見て −counter_z を下回るなら新規を出さない
                    (流れに逆らう抜け)
    max_held        口座が持つペアの数(この足で新しく建てる分も含む)をこれまでにする
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from .. import STRATEGIES
from .carry_breakout import MainCarryBreakout


class _CapContext:
    """新規の数を max_held までに抑える ctx(手仕舞い・決済はそのまま通す)。"""

    def __init__(self, owner: MainCarryGate, ctx):
        self._o = owner
        self._ctx = ctx
        self.open = sum(1 for pair in owner.pairs if ctx.side(pair) != 0)

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def target(self, pair, side, units=None, sl=None):
        cur = self._ctx.side(pair)
        if side == 0:
            if cur != 0:
                self.open -= 1
        elif cur == 0:
            if self.open >= self._o.p.max_held:
                self._o.blocked["held"] += 1
                return
            self.open += 1
        self._ctx.target(pair, side, units, sl)


@STRATEGIES.register("main_carry_gate", version="1",
                     description="本命: main_carry_breakout に新規の関門"
                                 "(荒い・流れに逆らう抜け・同時に持つ数)を足したもの")
class MainCarryGate(MainCarryBreakout):
    name = "main_carry_gate"
    tier: ClassVar[str] = "main"

    @dataclass
    class Params(MainCarryBreakout.Params):
        entry_vol_max: float = 0.0      # 0 = 見ない
        vol_window: int = 500
        counter_z: float = 0.0          # 0 = 見ない
        max_held: int = 0               # 0 = 見ない

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        self.blocked.update({"gate_vol": 0, "gate_counter": 0, "held": 0})
        self.vol_rel, self.trend_z = {}, {}
        for pair in self.pairs:
            a = self.atr[pair]
            typical = pd.Series(a).shift(1).rolling(p.vol_window, min_periods=p.vol_window).median().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                self.vol_rel[pair] = a / typical
            lc = np.log(self.c[pair])
            r = np.diff(lc, prepend=lc[0])
            vol20 = pd.Series(r).rolling(20, min_periods=20).std(ddof=0).to_numpy()
            z = np.full(len(lc), np.nan)
            z[60:] = (lc[60:] - lc[:-60]) / np.maximum(vol20[60:] * np.sqrt(60), 1e-12)
            self.trend_z[pair] = z

    def allowed(self, pair: str, i: int) -> bool:
        p = self.p
        if p.entry_vol_max > 0:
            v = self.vol_rel[pair][i]
            if np.isfinite(v) and v > p.entry_vol_max:
                self.blocked["gate_vol"] += 1
                return False
        if p.counter_z > 0:
            side = self._want(pair, i)
            z = self.trend_z[pair][i]
            if side != 0 and np.isfinite(z) and z * side < -p.counter_z:
                self.blocked["gate_counter"] += 1
                return False
        return super().allowed(pair, i)

    def on_bar(self, i, ctx):
        if self.p.max_held > 0:
            ctx = _CapContext(self, ctx)
        super().on_bar(i, ctx)
