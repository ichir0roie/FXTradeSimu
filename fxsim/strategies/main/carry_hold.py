"""S046 main_carry_hold / S047 main_carry_short: スワップ狙いを「持ち続ける型」と「数日で出る型」に分けたもの(E047)。

S046 main_carry_hold  金利差の順位で上位 top_n ペアを常に持つ
    順位      受け取る向きの金利差(業者の取り分を引いた後)が min_carry 以上のペアを、rank_by(carry = 金利差 /
              c2r = キャリー・トゥ・リスク)の高い順に並べる。毎日決め直す
    新規      上位 top_n に入っていて、持っていなければ入る。lt_n > 0 なら、終値が lt_n 日前より受け取る向きに
              動いているペアだけ(長期的にスワップの向きに動いていないペアには入らない)
    手仕舞い  順位から外れたペアは、exit = "profit_or_stop" なら値動き + スワップの合計が 0 以上になったとき、または
              終値が建値から建てた足の ATR x stop_atr 逆行したときに閉じる。"now" なら外れたらすぐ閉じる。
              金利差の向きが変わったらすぐ閉じる(S040 と同じ)

S047 main_carry_short  受け取る向きの信号で入り、hold 本で出る
    信号      signal = break(終値が n_in 日の高値抜け)/ macd(MACD がシグナルを抜く)/ either(どちらか)
    手仕舞い  建ててから hold 本。extend なら、持っている間に信号がまた点いたら、そこから数え直す
    下限      S040 と同じ(min_carry・c2r_min)

急変用の逆指値・口座の床・証拠金の上限・スプレッドの門・量は S040 のまま。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..indicators import macd
from .carry_breakout import MainCarryBreakout


class _WithContext(MainCarryBreakout):
    """信号の中から建玉を見るために、足ごとの ctx を覚えておく。"""

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        self._ctx = None

    def on_bar(self, i, ctx):
        self._ctx = ctx
        super().on_bar(i, ctx)

    def _pos(self, pair: str):
        return self._ctx._pb.legs[pair].pos if self._ctx is not None else None

    def _held_bars(self, pair: str, i: int) -> int:
        t0 = self.open_pairs.get(pair)
        return i - int(np.searchsorted(self.t, t0)) if t0 is not None else 0


@STRATEGIES.register("main_carry_hold", version="1",
                     description="本命: 10 ペアの口座で、金利差の順位の上位を受け取る向きで常に持つ")
class MainCarryHold(_WithContext):
    name = "main_carry_hold"

    @dataclass
    class Params(MainCarryBreakout.Params):
        min_carry: float = 0.0
        c2r_min: float = 0.0
        units: float = 1000.0
        top_n: int = 5
        rank_by: str = "carry"          # carry / c2r
        lt_n: int = 250                 # 0 = 見ない
        exit: str = "profit_or_stop"    # profit_or_stop / now
        stop_atr: float = 3.0

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        if p.rank_by not in ("carry", "c2r") or p.exit not in ("profit_or_stop", "now"):
            raise ValueError("rank_by must be carry / c2r, exit must be profit_or_stop / now")
        self.lt = {}
        for pair, b in bars_by_pair.items():
            d = np.zeros(len(b.close))
            if p.lt_n > 0:
                d[p.lt_n:] = np.sign(b.close[p.lt_n:] - b.close[:-p.lt_n])
            self.lt[pair] = d.astype(int)
        self.warmup = max(self.warmup, p.lt_n + 1)
        self.top: set[str] = set()

    def rank_bar(self, i: int) -> None:
        p = self.p
        score = self.carry if p.rank_by == "carry" else self.c2r
        cands = [(float(score[pair][i]), pair) for pair in self.pairs
                 if self.side[pair][i] != 0 and self.carry[pair][i] >= max(p.min_carry, 1e-9)
                 and np.isfinite(score[pair][i])]
        cands.sort(key=lambda x: (-x[0], x[1]))
        self.top = {pair for _, pair in cands[:p.top_n]}

    def signal_in(self, pair: str, i: int, side: int) -> bool:
        if pair not in self.top:
            return False
        return self.p.lt_n <= 0 or int(self.lt[pair][i]) == side

    def signal_out(self, pair: str, i: int, cur: int) -> bool:
        if pair in self.top:
            return False
        if self.p.exit == "now":
            return True
        pos = self._pos(pair)
        if pos is None:
            return True
        move = (self.c[pair][i] - pos.entry_price) * pos.side
        if move * pos.units + pos.swap >= 0:
            return True
        a = self.entry_atr.get(pair, np.nan)
        return bool(np.isfinite(a) and a > 0 and move <= -self.p.stop_atr * a)


@STRATEGIES.register("main_carry_short", version="1",
                     description="本命: 10 ペアの口座で、受け取る向きの信号で入り、数日で出る")
class MainCarryShort(_WithContext):
    name = "main_carry_short"

    @dataclass
    class Params(MainCarryBreakout.Params):
        n_in: int = 10
        n_out: int = 0
        min_carry: float = 0.02
        c2r_min: float = 0.4
        signal: str = "break"           # break / macd / either
        hold: int = 3
        extend: bool = True

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        if p.signal not in ("break", "macd", "either"):
            raise ValueError("signal must be break / macd / either")
        self.cross = {}
        for pair, b in bars_by_pair.items():
            m, s = macd(b.close)
            st = np.nan_to_num(np.sign(m - s)).astype(int)
            self.cross[pair] = np.where(st != np.r_[0, st[:-1]], st, 0)     # 抜けた足だけ、抜けた向き
        self.warmup = max(self.warmup, 100)
        self.last_sig: dict[str, int] = {}

    def _signal(self, pair: str, i: int, side: int) -> bool:
        p = self.p
        hit = False
        if p.signal in ("break", "either"):
            hit |= MainCarryBreakout.signal_in(self, pair, i, side)
        if p.signal in ("macd", "either"):
            hit |= int(self.cross[pair][i]) == side
        return bool(hit)

    def signal_in(self, pair: str, i: int, side: int) -> bool:
        hit = self._signal(pair, i, side)
        if hit:
            self.last_sig[pair] = i
        return hit

    def signal_out(self, pair: str, i: int, cur: int) -> bool:
        if self.p.extend and self._signal(pair, i, cur):
            self.last_sig[pair] = i
            return False
        start = i - self._held_bars(pair, i)
        if self.p.extend:
            start = max(start, self.last_sig.get(pair, start))
        return i - start >= self.p.hold


__all__ = ["MainCarryHold", "MainCarryShort"]
