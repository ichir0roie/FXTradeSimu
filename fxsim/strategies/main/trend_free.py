"""S045 main_trend_free: 本命 3 本目(S040)から「スワップを受け取る向きだけ」を外せるようにした対照。

    sides     recv = 受け取る向きだけ(S040 と同じ向き。金利差の下限は min_carry)/ both = 金利差を見ずに信号の向き /
              pay = 信号が払う向きのときだけ(対照)
    signal    break = 終値が n_in 日の高値・安値を抜けたら入り、反対側の n_out 日のチャネルで出る /
              ema   = 速い EMA と遅い EMA の向きで入り、返ったら出る(signal_every 本ごとに判定)/
              mom   = mom_n 日前の終値より上か下かで入り、返ったら出る(signal_every 本ごとに判定)/
              pullback = 終値が pb_slow 日平均より上で RSI(pb_rsi) が pb_enter を割ったら買い(売りは鏡)、
                         RSI が pb_exit を越えるか pb_max 本で出る(S048 の 10 ペアの口座版、E048)
both・pay では金利差の下限とキャリー・トゥ・リスクを使わない(c2r_min は 0 にする)。
エンジンの swap_side_only も外して回す。
急変用の逆指値・口座の床・証拠金の上限・スプレッドの門・量は S040 のまま。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..indicators import ema, rsi, sma
from .carry_breakout import MainCarryBreakout


@STRATEGIES.register("main_trend_free", version="1",
                     description="対照: 10 ペアの口座で、スワップの向きを見ずに(または払う向きだけで)"
                                 "高値抜け・EMA・モメンタムの向きに持つ")
class MainTrendFree(MainCarryBreakout):
    name = "main_trend_free"

    @dataclass
    class Params(MainCarryBreakout.Params):
        n_in: int = 10
        n_out: int = 5
        sides: str = "both"             # recv / both / pay
        signal: str = "break"           # break / ema / mom
        ema_fast: int = 5
        ema_slow: int = 20
        mom_n: int = 63
        signal_every: int = 1
        pb_slow: int = 200
        pb_rsi: int = 2
        pb_enter: float = 10.0
        pb_exit: float = 50.0
        pb_max: int = 10

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        if p.sides not in ("recv", "both", "pay") or p.signal not in ("break", "ema", "mom", "pullback"):
            raise ValueError("sides must be recv / both / pay, signal must be break / ema / mom / pullback")
        self.dir = {}
        for pair, b in bars_by_pair.items():
            c = b.close
            if p.signal == "break":
                d = np.where(c > self.hi_in[pair], 1, np.where(c < self.lo_in[pair], -1, 0))
            elif p.signal == "ema":
                d = np.nan_to_num(np.sign(ema(c, p.ema_fast) - ema(c, p.ema_slow)))
            elif p.signal == "pullback":
                d = self._pullback(c)
            else:
                d = np.zeros(len(c))
                d[p.mom_n:] = np.sign(c[p.mom_n:] - c[:-p.mom_n])
            self.dir[pair] = d.astype(int)
        extra = {"break": 0, "ema": 3 * p.ema_slow, "mom": p.mom_n + 1, "pullback": p.pb_slow}[p.signal]
        self.warmup = max(self.warmup, extra)
        self._ctx = None

    def _pullback(self, c: np.ndarray) -> np.ndarray:
        """押し目の建玉の状態(+1 / −1 / 0)。入りの足から、RSI が戻るか pb_max 本たつまで同じ向き。"""
        p = self.p
        ma, r = sma(c, p.pb_slow), rsi(c, p.pb_rsi)
        ok = np.isfinite(ma) & np.isfinite(r)
        entry = np.where(ok & (c > ma) & (r < p.pb_enter), 1, np.where(ok & (c < ma) & (r > 100 - p.pb_enter), -1, 0))
        out = np.zeros(len(c), dtype=int)
        cur, age = 0, 0
        for i in range(len(c)):
            if cur != 0:
                age += 1
                if (r[i] > p.pb_exit if cur > 0 else r[i] < 100 - p.pb_exit) or age >= p.pb_max:
                    cur = 0
            if cur == 0 and entry[i] != 0:
                cur, age = int(entry[i]), 0
            out[i] = cur
        return out

    def on_bar(self, i, ctx):
        self._ctx = ctx
        super().on_bar(i, ctx)

    def _tick(self, i: int) -> bool:
        return self.p.signal == "break" or i % max(1, int(self.p.signal_every)) == 0

    def _want(self, pair: str, i: int) -> int:
        p = self.p
        cur = self._ctx.side(pair) if self._ctx is not None else 0
        recv = int(self.side[pair][i])
        if cur != 0:                     # 持っている間は、金利差の向きが変わっても信号の出まで持つ(recv は S040 と同じ)
            if p.sides == "recv" and (recv != cur or self.carry[pair][i] < p.min_carry):
                return 0
            return cur
        if not self._tick(i):
            return 0
        d = int(self.dir[pair][i])
        if p.sides == "recv":
            return d if d == recv and self.carry[pair][i] >= p.min_carry else 0
        if p.sides == "pay":
            return d if recv != 0 and d == -recv else 0
        return d

    def signal_in(self, pair: str, i: int, side: int) -> bool:
        return side != 0

    def signal_out(self, pair: str, i: int, cur: int) -> bool:
        if self.p.signal == "break":
            return super().signal_out(pair, i, cur)
        return self._tick(i) and int(self.dir[pair][i]) != cur


__all__ = ["MainTrendFree"]
