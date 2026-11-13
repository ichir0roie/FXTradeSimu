"""S902 basket_timing_null: 複数ペアの戦略の「時期だけでたらめ版」(受け取る向きだけ建てる口座の検証の帰無仮説)。

swap_side_only(スワップを払う向きの新規は出さない、E022)の口座では、戦略が選べるのは「いつ・どのペアを・どれだけの間持つか」
だけで、向きはペアの金利差で決まる。方向でたらめ版(basket_direction_null)は払う向きを引くと注文が弾かれ、中の戦略が次の足で
入り直すたびに引き直すので、本体とほぼ同じ取引になる(E032: B049 は実データで 64% が同じ取引)。これでは帰無にならない。

この帰無は、最初の足で中の戦略を同じ相場・同じ設定で回し(PortfolioBacktester.replicate)、取引をペアごとの
「持った本数・量」の束にして、時期だけを乱数で置き直す:
    - 同じペア・同じ本数・同じ量。向きは建てる足の受け取る向き(エンジンが受け付ける足だけに置く)
    - 置ける足は、判断の足で、新規を出せて(取引時間の内)、受け取りがプラス(業者の取り分を引いて)の足。
      中の戦略の下限(金利差の大きさなど)は使わない(下限も戦略の情報の一部)。同じペアの中では重ならない
    - 同時に持つ数は、中の戦略が同時に持った最大の数まで。超える足、受け取りが無くなった足では
      新規を遅らせる(本数は変えない)
    - 損切りは置かない(中の戦略の急変用の逆指値の幅は取引に残らない)。決めた本数で手仕舞う
同じ相場パターンで比べた損益の差が、中の戦略の「いつ・どのペアを持つか」を選ぶ力になる。
口座が swap_side_only でなければ、置ける足は判断の足すべて、向きは取引ごとに乱数(方向と時期の両方がでたらめ)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .. import STRATEGIES
from ..basket import BasketStrategy


@STRATEGIES.register("basket_timing_null", version="1",
                     description="複数ペアの戦略と同じペア・持つ本数・量で、時期だけを乱数にしたもの"
                                 "(受け取る向きだけの口座の検証の帰無仮説)")
class BasketTimingNull(BasketStrategy):
    name = "basket_timing_null"
    # 最初の足で中の戦略を全期間回すので、作りとして因果的でない(check-lookahead から外す)。帰無が使う未来の情報は
    # 「持つ本数・量」と「受け取れる足か」だけで、値動きの向きは使わず、時期は乱数なので未来の値動きとそろわない
    lookahead_exempt: ClassVar[str] = "帰無: 中の戦略の持ち方(本数・量)だけを使い、時期は乱数(値動きの向きは使わない)"

    @dataclass
    class Params:
        inner: str = "xs_carry"
        inner_params: dict = field(default_factory=dict)
        timeframe: int = 1440

    def __init__(self, **params):
        super().__init__(**params)
        self.warmup = 0

    def _inner(self):
        return STRATEGIES.create(self.p.inner, {**self.p.inner_params, "timeframe": self.p.timeframe})

    def prepare(self, bars_by_pair, rng):
        self.rng = np.random.default_rng(rng.integers(0, 2**63))
        self.planned = False
        self.queue: dict[str, list[tuple[int, int, float, int]]] = {}
        self.open: dict[str, tuple[int, int]] = {}
        self.stats = {"planned": 0, "missed": 0, "entered": 0, "delayed": 0, "max_open": 0}

    def _plan(self, pb) -> None:
        ref = pb.replicate(self._inner())
        ref.run()
        n = len(pb.bars)
        t = pb.bars.t
        idx = {int(x): k for k, x in enumerate(t)}
        events = []
        books: dict[str, list[tuple[int, float]]] = {}
        for tr in ref.trades:
            a = idx.get(int(tr.entry_t))
            L = max(1, int(tr.bars_held))
            if a is None:
                continue
            books.setdefault(tr.pair, []).append((L, float(tr.units)))
            events += [(a, 1), (a + L, -1)]
        cur = mx = 0
        for _, d in sorted(events, key=lambda e: (e[0], e[1])):
            cur += d
            mx = max(mx, cur)
        self.max_open = mx
        warmup = int(getattr(ref.strategy, "warmup", 0))
        swap_only = bool(pb.cfg.swap_side_only)
        for pair, book in books.items():
            leg = pb.legs[pair]
            ok = pb.decide[:n] & pb.can_enter[:n]
            ok[:warmup] = False
            ok[n - 2:] = False
            if swap_only:
                ok &= np.array([_recv_side(leg, k) != 0 for k in range(n)])
            cand = np.flatnonzero(ok)
            busy = np.zeros(n + 2, dtype=bool)
            out = []
            for j in self.rng.permutation(len(book)):
                L, units = book[j]
                for _ in range(300):
                    if not len(cand):
                        break
                    d = int(cand[self.rng.integers(len(cand))])
                    if d + L + 1 < n and not busy[d:d + L + 2].any():
                        busy[d:d + L + 2] = True
                        side = 0 if swap_only else (1 if self.rng.random() < 0.5 else -1)
                        out.append((d, L, units, side))
                        break
                else:
                    self.stats["missed"] += 1
            self.queue[pair] = sorted(out)
            self.stats["planned"] += len(out)
        self.planned = True

    def _side(self, leg, i: int, planned: int) -> int:
        return planned or _recv_side(leg, i)

    def on_bar(self, i, ctx):
        pb = ctx._pb
        if not self.planned:
            self._plan(pb)
        for pair in list(self.open):
            exit_i, entry_i = self.open[pair]
            if ctx.side(pair) == 0 and i > entry_i + 1:          # 弾かれた注文・ロスカット
                del self.open[pair]
            elif i >= exit_i:
                ctx.target(pair, 0)
                del self.open[pair]
        for pair, q in self.queue.items():
            if not q or q[0][0] > i or pair in self.open or ctx.side(pair) != 0:
                continue
            side = self._side(pb.legs[pair], i, q[0][3])
            if side == 0 or not ctx.can_enter or len(self.open) >= self.max_open:
                self.stats["delayed"] += 1
                continue
            _, L, units, _ = q.pop(0)
            ctx.target(pair, side, units)
            self.open[pair] = (i + L, i)
            self.stats["entered"] += 1
            self.stats["max_open"] = max(self.stats["max_open"], len(self.open))


def _recv_side(leg, k: int) -> int:
    """足 k でスワップを受け取る向き(エンジンが受け付けなければ 0)。"""
    diff = getattr(leg, "rate_diff", None)
    s = (int(np.sign(diff[k])) if diff is not None else (1 if leg.swap_side_ok(1, k) else -1)) or 1
    return s if leg.swap_side_ok(s, k) else 0
