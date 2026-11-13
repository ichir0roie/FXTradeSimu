"""S044 main_carry_ema: 本命 3 本目(S040)の入りと出を、EMA の向きに替えた版。

S040 との違い
    シグナル    entry = "ema" なら、速い EMA(ema_fast)が遅い EMA(ema_slow)より受け取る向きの側にある間は入る
                (持っていなければ、交差した日でなくても入る)。"break" なら S040 と同じ n_in 日の高値抜け
    手仕舞い    速い EMA が遅い EMA の反対側に回ったら出る(n_out のチャネルは使わない)。
                min_hold > 0 なら、建ててからその本数が経つまでは EMA では出ない
    判断の間隔  signal_every > 1 なら、EMA の入りと出はその本数ごとの足でだけ判定する(5 = 週に 1 回)
金利差の下限・キャリー・トゥ・リスク・受け取る向きだけ・急変用の逆指値・口座の床・証拠金の上限・スプレッドの門は
S040 のまま。金利差が下限を割る・向きが変わる・口座の床は、min_hold・signal_every に関係なく毎日効く。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..indicators import ema
from .carry_breakout import MainCarryBreakout


@STRATEGIES.register("main_carry_ema", version="1",
                     description="本命: 10 ペアの口座でスワップを受け取る向きだけ、EMA の向きで入り"
                                 "EMA が返ったら出る")
class MainCarryEma(MainCarryBreakout):
    name = "main_carry_ema"

    @dataclass
    class Params(MainCarryBreakout.Params):
        n_in: int = 10
        n_out: int = 5                  # EMA で出るので使わない(利確の後の入り直しの判定にだけ残る)
        min_carry: float = 0.02
        c2r_min: float = 0.4
        ema_fast: int = 10
        ema_slow: int = 30
        entry: str = "ema"              # ema(EMA の向き)/ break(n_in 日の高値抜け)
        min_hold: int = 0               # 建ててからこの本数が経つまでは EMA では出ない
        signal_every: int = 1           # EMA の入りと出を判定する間隔(本)

    def prepare(self, bars_by_pair, rng):
        super().prepare(bars_by_pair, rng)
        p = self.p
        if p.entry not in ("ema", "break"):
            raise ValueError("entry must be 'ema' or 'break'")
        self.trend = {pair: np.nan_to_num(np.sign(ema(b.close, p.ema_fast) - ema(b.close, p.ema_slow))).astype(int)
                      for pair, b in bars_by_pair.items()}
        self.warmup = max(self.warmup, 3 * p.ema_slow)

    def _tick(self, i: int) -> bool:
        return i % max(1, int(self.p.signal_every)) == 0

    def signal_in(self, pair: str, i: int, side: int) -> bool:
        if self.p.entry == "break":
            return super().signal_in(pair, i, side)
        return self._tick(i) and int(self.trend[pair][i]) == side

    def signal_out(self, pair: str, i: int, cur: int) -> bool:
        if not self._tick(i) or int(self.trend[pair][i]) == cur:
            return False
        t0 = self.open_pairs.get(pair)
        if self.p.min_hold > 0 and t0 is not None:
            held = i - int(np.searchsorted(self.t, t0))
            if held < self.p.min_hold:
                return False
        return True


__all__ = ["MainCarryEma"]
