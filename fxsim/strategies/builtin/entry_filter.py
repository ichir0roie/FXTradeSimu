"""S042 entry_filter: 別の戦略を包み、相場の状態で新規を止める(E035)。

中の戦略(inner)はそのまま動かし、新規(buy / sell / add)を出す足の状態が条件に掛かったら新規だけを止める。
止めた足でドテンしようとしたときは、反対の建玉の決済だけを行う(エンジンの取引時間外と同じ扱い)。
決済・損切りの付け替えは止めない。時間帯・曜日で止めるのは動作ケースの schedule(backtest/schedule.py)を使う。

    荒い      ATR(vol_n) / 直前 vol_window 本の ATR の中央値 > max_vol_ratio なら止める(0 = 見ない)
    一方向    er_n 本の効率比(|er_n 本の動き| / 1 本ごとの動きの絶対値の和)> max_er なら止める(0 = 見ない)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import atr


class _FilterContext:
    def __init__(self, owner: EntryFilter, ctx):
        self._o = owner
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def _enter(self, side: int, **kw) -> None:
        if self._o.blocked[self._o.i]:
            self._o.n_blocked += 1
            if self._ctx.side == -side:
                self._ctx.close()
            return
        (self._ctx.buy if side > 0 else self._ctx.sell)(**kw)

    def buy(self, **kw) -> None:
        self._enter(1, **kw)

    def sell(self, **kw) -> None:
        self._enter(-1, **kw)

    def add(self, *a, **kw) -> None:
        if not self._o.blocked[self._o.i]:
            self._ctx.add(*a, **kw)


@STRATEGIES.register("entry_filter", version="1",
                     description="別の戦略を包み、荒い局面・一方向の局面では新規を出さない")
class EntryFilter(Strategy):
    name = "entry_filter"

    @dataclass
    class Params:
        inner: str = "ma_cross"
        inner_params: dict = field(default_factory=dict)
        timeframe: int = 60
        max_vol_ratio: float = 0.0
        vol_n: int = 14
        vol_window: int = 500
        max_er: float = 0.0
        er_n: int = 20

    def __init__(self, **params):
        super().__init__(**params)
        inner_params = {**self.p.inner_params, "timeframe": self.p.timeframe}
        self.inner = STRATEGIES.create(self.p.inner, inner_params)
        self.i = 0
        self.n_blocked = 0

    def prepare(self, bars, rng):
        self.inner.prepare(bars, rng)
        self.warmup = getattr(self.inner, "warmup", 0)
        p = self.p
        n = len(bars.close)
        blocked = np.zeros(n, dtype=bool)
        if p.max_vol_ratio > 0:
            a = atr(bars.high, bars.low, bars.close, p.vol_n)
            typical = pd.Series(a).shift(1).rolling(p.vol_window, min_periods=p.vol_window).median().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                blocked |= np.nan_to_num(a / typical, nan=0.0) > p.max_vol_ratio
        if p.max_er > 0:
            lc = np.log(bars.close)
            cs = np.cumsum(np.abs(np.diff(lc, prepend=lc[0])))
            er = np.zeros(n)
            k = p.er_n
            er[k:] = np.abs(lc[k:] - lc[:-k]) / np.maximum(cs[k:] - cs[:-k], 1e-12)
            blocked |= er > p.max_er
        self.blocked = blocked

    def on_bar(self, i, ctx):
        self.i = i
        self.inner.on_bar(i, _FilterContext(self, ctx))
