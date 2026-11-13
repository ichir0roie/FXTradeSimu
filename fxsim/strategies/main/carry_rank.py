"""S041 main_carry_rank: 本命 3 本目(S040)の、ペアを順位で選び量に差を付ける版。

S040 との違い
    選び方      安定面の条件(受け取りの年率 ≥ min_carry、rank_vol_n 日ボラの年率 ≤ max_vol、
                スプレッドを受け取りで取り返す日数 ≤ max_spread_days)を満たすペアを、
                キャリー・トゥ・リスク(受け取りの年率 / rank_vol_n 日ボラの年率)の高い順に並べる。
                順位は毎日の足で決め直す(その日までの値だけ。後から見た上位を固定しない)
    量          上から weights の割合。1 位は unit x weights[0] 通貨。weights の数より下の順位には新規を出さない
    外し方      drop_rank > 0 なら、持っているペアの順位が drop_rank 位より下(または条件の外)になったら手仕舞う。
                0 なら S040 と同じ(反対側のチャネル・金利差が下限を割る)
入り(ブレイクアウト)、急変用の逆指値、口座の床、証拠金の上限、スプレッドの門は S040 のまま。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import STRATEGIES
from .carry_breakout import MainCarryBreakout


@STRATEGIES.register("main_carry_rank", version="1",
                     description="本命: 10 ペアから金利差とボラの釣り合いで上位を選び、順位の割合で量を変える"
                                 "スワップ狙い + ブレイクアウト")
class MainCarryRank(MainCarryBreakout):
    name = "main_carry_rank"

    @dataclass
    class Params(MainCarryBreakout.Params):
        n_in: int = 20
        n_out: int = 10
        c2r_min: float = 0.0            # 順位で選ぶので、S040 の下限は既定で使わない
        weights: list = field(default_factory=lambda: [4, 3, 2, 1])
        unit: float = 1000.0            # weights の 1 あたりの通貨
        rank_vol_n: int = 250           # 順位に使うボラの日数(長いほど落ち着いた順位)
        max_vol: float = 0.15           # 0 = 見ない
        max_spread_days: float = 10.0   # 0 = 見ない。典型のスプレッド / 1 日の受け取り
        drop_rank: int = 0

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        self.vol_rank, self.spread_days = {}, {}
        for pair, b in bars_by_pair.items():
            r = np.diff(np.log(b.close), prepend=np.log(b.close[0]))
            vol = np.full(len(r), np.nan)
            for i in range(p.rank_vol_n, len(r)):
                vol[i] = r[i - p.rank_vol_n + 1:i + 1].std() * np.sqrt(260)
            self.vol_rank[pair] = vol
            with np.errstate(divide="ignore", invalid="ignore"):
                daily = self.carry[pair] * b.close / 365.0
                self.spread_days[pair] = np.where(daily > 0, self.typical[pair] / daily, np.inf)
        self.warmup = max(self.warmup, p.rank_vol_n + 1)
        self.rank: dict[str, int] = {}
        self.rank_hist: dict[str, int] = {pair: 0 for pair in self.pairs}   # 1 位〜の日数(結果の読み取り用)

    def stable(self, pair: str, i: int) -> bool:
        p = self.p
        v = self.vol_rank[pair][i]
        if not (np.isfinite(v) and v > 0) or self.carry[pair][i] < p.min_carry or self.side[pair][i] == 0:
            return False
        if p.max_vol > 0 and v > p.max_vol:
            return False
        sd = self.spread_days[pair][i]
        return not (p.max_spread_days > 0 and np.isfinite(self.typical[pair][i]) and sd > p.max_spread_days)

    def rank_bar(self, i: int) -> None:
        score = [(-self.carry[pair][i] / self.vol_rank[pair][i], pair) for pair in self.pairs if self.stable(pair, i)]
        self.rank = {pair: k for k, (_, pair) in enumerate(sorted(score))}
        for pair, k in self.rank.items():
            if k == 0:
                self.rank_hist[pair] += 1

    def drop(self, pair: str, i: int) -> bool:
        n = self.p.drop_rank
        return n > 0 and self.rank.get(pair, n) >= n

    def allowed(self, pair: str, i: int) -> bool:
        k = self.rank.get(pair)
        return k is not None and k < len(self.p.weights) and self.p.weights[k] > 0

    def entry_units(self, pair: str, i: int) -> float:
        return self.p.unit * float(self.p.weights[self.rank[pair]])


__all__ = ["MainCarryRank"]
