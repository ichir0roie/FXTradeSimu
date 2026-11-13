"""複数ペアの戦略の基底。

書き方(1 ペアの Strategy と同じ約束):
    1. multi = True。prepare(bars_by_pair, rng) は bars_by_pair(ペア -> Bars、全ペア同じ時刻)で因果的な計算だけ
    2. on_bar(i, ctx) で足 i の終値時点の判断をする。ctx は backtest.portfolio.PortfolioContext。
       ctx.target(ペア, 向き, units) で持ち高を決める(次の足の始値で約定)
    3. 1 ペアの相場(MarketData に legs が無い)では回らない(実験では飛ばす)

CrossSectional は「ペアの点数で順位を付け、上位 n_long を買い、下位 n_short を売る」を hold_days 本ごとに行う。
対円のペアだけで組むので、買いと売りの数が同じなら円そのものの上げ下げは打ち消し合い、通貨どうしの差だけが残る。
1 ペアの建玉の金額 = 初期資金 x gross ÷ (n_long + n_short)。持ち続けるペアの量は入れ替えのときも変えない(回転を抑える)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..data import Bars
from ..data.rates import MULTI_RATES, load_rates
from .base import Strategy


class BasketStrategy(Strategy):
    multi: ClassVar[bool] = True

    @dataclass
    class Params:
        timeframe: int = 1440

    def prepare(self, bars_by_pair: dict[str, Bars], rng: np.random.Generator) -> None:  # pragma: no cover
        raise NotImplementedError


def closes(bars_by_pair: dict[str, Bars]) -> tuple[list[str], np.ndarray]:
    pairs = list(bars_by_pair)
    return pairs, np.column_stack([bars_by_pair[p].close for p in pairs])


def lagged_log_return(c: np.ndarray, lookback: int, skip: int = 0) -> np.ndarray:
    """足 i で、log(c[i - skip] / c[i - lookback])。足りない足は nan。"""
    out = np.full(c.shape, np.nan)
    if lookback < len(c):
        lc = np.log(c)
        out[lookback:] = lc[lookback - skip:len(c) - skip] - lc[:len(c) - lookback]
    return out


def log_returns(c: np.ndarray) -> np.ndarray:
    """(足 x ペア)の 1 本ごとの log リターン。先頭の足は nan。"""
    return np.vstack([np.full((1, c.shape[1]), np.nan), np.diff(np.log(c), axis=0)])


def rolling_vol(c: np.ndarray, window: int, per_year: float = 260.0) -> np.ndarray:
    """足 i までの window 本の log リターンの標準偏差(年率)。足りない足は nan。"""
    r = log_returns(c)
    vol = np.full(c.shape, np.nan)
    for i in range(window, len(c)):
        vol[i] = r[i - window + 1:i + 1].std(axis=0) * np.sqrt(per_year)
    return vol


def every(n: int, first: int, step: int) -> np.ndarray:
    """足 first から step 本ごとに True(入れ替えの足)。"""
    out = np.zeros(n, dtype=bool)
    out[first::max(1, int(step))] = True
    return out


def rate_differentials(pairs: list[str], t: np.ndarray, rates_path: str = MULTI_RATES) -> np.ndarray:
    rt = load_rates(rates_path)
    return np.column_stack([rt.pair_differential(p, t) for p in pairs])


def rank_sides(score: np.ndarray, n_long: int, n_short: int) -> np.ndarray:
    """1 本の足の点数(ペアごと、nan は対象外)から、向き(+1 / -1 / 0)。同点はペアの並び順(安定ソート)。"""
    sides = np.zeros(len(score), dtype=int)
    ok = np.flatnonzero(np.isfinite(score))
    if len(ok) < n_long + n_short or len(ok) == 0:
        return sides
    order = ok[np.argsort(-score[ok], kind="stable")]
    if n_long:
        sides[order[:n_long]] = 1
    if n_short:
        sides[order[len(order) - n_short:]] = -1
    return sides


class CrossSectional(BasketStrategy):
    """点数の順位で買い・売りを決める。サブクラスは score(prepare の中で呼ぶ)を書く。"""

    @dataclass
    class Params:
        timeframe: int = 1440
        hold_days: int = 21          # 入れ替えの間隔(本)
        n_long: int = 2
        n_short: int = 2
        gross: float = 1.5           # 建玉の総額 / 初期資金

    def score(self, bars_by_pair: dict[str, Bars], c: np.ndarray) -> np.ndarray:  # pragma: no cover
        """(足 x ペア)の点数。大きいほど買いたい。その足の終値までで計算できない所は nan。"""
        raise NotImplementedError

    def prepare(self, bars_by_pair, rng):
        self.pairs, c = closes(bars_by_pair)
        self.c = c
        s = self.score(bars_by_pair, c)
        ready = np.flatnonzero(np.isfinite(s).sum(axis=1) >= self.p.n_long + self.p.n_short)
        first = int(ready[0]) if len(ready) else len(c)
        self.warmup = first
        self.rebalance = every(len(c), first, self.p.hold_days)
        self.sides = np.zeros(c.shape, dtype=int)
        for i in np.flatnonzero(self.rebalance):
            self.sides[i] = rank_sides(s[i], self.p.n_long, self.p.n_short)
        self.scores = s

    def leg_units(self, ctx, j: int, i: int) -> float:
        n = max(1, self.p.n_long + self.p.n_short)
        notional = ctx.initial_balance * self.p.gross / n
        return float(np.floor(notional / self.c[i, j] / 1000.0) * 1000.0)

    def on_bar(self, i, ctx):
        if not self.rebalance[i]:
            return
        for j, pair in enumerate(self.pairs):
            ctx.target(pair, int(self.sides[i, j]), self.leg_units(ctx, j, i))


__all__ = ["MULTI_RATES", "BasketStrategy", "CrossSectional", "closes", "every", "lagged_log_return", "log_returns",
           "rank_sides", "rate_differentials", "rolling_vol"]
