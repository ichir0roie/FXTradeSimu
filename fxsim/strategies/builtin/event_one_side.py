"""S053 event_one_side: 大きく動く時を狙い、片方の向きにだけ入る。

跳ねたら利確、逆指値は置かず、負けは落ち着いてから閉じる。

利用者の案(2026-11-13)をそのまま本物のエンジンに載せたもの(研究の約定では E062・E063 で不合格)。1 時間足で動かす。
    trigger = nfp    米雇用統計(E062 の暦の決まり、8:30 ニューヨーク)の足の始値で入る(その前の足の終値で決める)
    trigger = claims 米新規失業保険申請件数(毎週木曜 8:30 ニューヨーク、E067)。入り方は nfp と同じ
    trigger = spike  幅が直近 480 本の中央値の spike_k 倍以上・実体が幅の半分以上の足が確定したら、次の足で入る(E063)
向き: 20 日(480 本)の値動きの向き(nfp・claims)/ 跳ねた向き(spike)。方向でたらめ版は pair-eval が自動で作る。
利確: 物差し x tp_mult(nfp は直近 12 回の発表の足の幅の中央値、spike は直近 480 本の幅の中央値)。逆指値は置かない。
手仕舞い: exit = calm なら、入った後の足の幅が直近 480 本の中央値以下に戻ったら閉じる。5d なら利確まで最長 120 本待つ。
どちらも最長 120 本で閉じる。
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ...calendar import claims_dates, nfp_dates
from ...calendar import nfp_bar_start as at_830
from .. import STRATEGIES
from ..base import Strategy

NS = 1_000_000_000


@STRATEGIES.register("event_one_side", version="2",
                     description="大きく動く時に片側だけ入り、利確のみ・逆指値なし、負けは落ち着いてから閉じる(E062・E063)")
class EventOneSide(Strategy):
    name = "event_one_side"

    @dataclass
    class Params:
        timeframe: int = 60
        trigger: str = "nfp"        # nfp / claims / spike
        tp_mult: float = 0.5
        exit: str = "calm"          # calm / 5d
        spike_k: float = 3.0
        max_bars: int = 120

    def prepare(self, bars, rng):
        n = len(bars)
        h, lo, c, o = bars.high, bars.low, bars.close, bars.open
        rng_ = h - lo
        self.med = pd.Series(rng_).rolling(480).median().shift(1).to_numpy(copy=True)   # 足 i の前の 480 本
        self.rng_ = rng_
        self.want = np.zeros(n, dtype=int)          # 足 i の終値で決める新規の向き
        self.scale = np.full(n, np.nan)             # 利確の物差し(価格)
        sec = bars.t // NS
        trend = np.zeros(n)
        trend[480:] = np.sign(c[480:] - c[:-480])
        if self.p.trigger in ("nfp", "claims"):
            first = pd.Timestamp(int(sec[0]), unit="s").date()
            last = pd.Timestamp(int(sec[-1]), unit="s").date()
            dates_fn = nfp_dates if self.p.trigger == "nfp" else claims_dates
            ev = {at_830(d) for d in dates_fn(date(first.year, first.month, 1), last)}
            idx = {int(s): k for k, s in enumerate(sec)}
            hist = []
            for s in sorted(ev):
                k = idx.get(s)
                if k is None:
                    continue
                if len(hist) >= 6 and k >= 481:
                    self.want[k - 1] = int(trend[k - 1]) or 1
                    self.scale[k - 1] = float(np.median(hist[-12:])) * o[k]
                hist.append(rng_[k] / o[k])
        else:
            mv = c - o
            ok = np.isfinite(self.med) & (np.abs(mv) >= 0.5 * rng_) & (rng_ >= self.p.spike_k * self.med)
            self.want[ok] = np.sign(mv[ok]).astype(int)
            self.scale[ok] = self.med[ok]
        self.warmup = 481
        self.entry_i = -1

    def on_bar(self, i, ctx):
        if ctx.side != 0:
            if self.entry_i < 0:
                self.entry_i = i
            held = i - self.entry_i
            calm = self.p.exit == "calm" and held >= 1 and self.rng_[i] <= self.med[i]
            if calm or held >= self.p.max_bars:
                ctx.close()
            return
        self.entry_i = -1
        w = int(self.want[i])
        if w == 0 or not np.isfinite(self.scale[i]):
            return
        tp = self.p.tp_mult * self.scale[i]
        if w > 0:
            ctx.buy(tp=tp)
        else:
            ctx.sell(tp=tp)
