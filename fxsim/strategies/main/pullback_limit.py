"""S049 main_pullback_limit: 日足の押し目(S048)を、指値で待って入る形にしたもの。1 時間足で動かす。

E050 で、押し目の損切り・利確・入り方の格子(336 通り)を 1 時間足の経路で回した。損切りと利確の幅は成績を変えず、
変えたのは入り方だった: 日足の終値から ATR x 0.5 だけ有利な値に指値を置いて 1 日待つと、成行の +2.3 bp / 取引が
+8.5〜10.3 bp になった(でたらめな入りの同じ格子との差 z 4.6)。
決まり(1 日 1 回、取引日が替わって decide_bars 本たった足の確定で判断する):
    手仕舞い  持っていて、前の取引日の RSI(rsi_n) が exit を越えた(売りは 100 − exit を割った)か、max_days 回の判断を
              過ぎたら成行で決済する
    新規      何も持っておらず、前の取引日の終値が slow 日の移動平均より上で RSI が enter を割っていたら買い(売りは鏡)。
              前の取引日の終値から ATR(atr_n) x limit_atr だけ下に買いの指値を置く(expire_hours で取り消し)。
              気配がすでに指値より有利なら成行。limit_atr = 0 なら常に成行(対照)
              vol_ratio_min > 0 なら、前の取引日の ATR(atr_n) / ATR(vol_long) がそれ以上の日だけ(荒い時期だけ、E051)
    損切り    建値から ATR x stop_atr の逆指値(サーバー)。量は、この損切りで有効証拠金の risk_per_trade を失う量
取引日はサーバー時刻(ニューヨーク + 7 時間)の日付で切る。夏時間・冬時間ともロールオーバーが 0 時になる。
日足の値は、その取引日の足が全部確定してから使う(先読みしない)。
日足未満の版(signal_hours < 24、E052): 信号の足を UTC の時刻で signal_hours ごとに切り、その足が確定するたびに判断する。
指値の期限は信号の足 1 本ぶん。ロールオーバーの前後に始まる足では新規を出さない。
rolling: 足の区切りの時刻(UTC 0・4・8…時)で成績が変わるので、区切りを 1 本ずつずらした全部の位相を持ち、
毎時「いま確定した位相」の信号で判断する。建玉は 1 つだけで、手仕舞いは建てた位相の足で見る。
方向でたらめ版(main_direction_null)では、新規の向きだけを乱数で決める(指値は鏡の位置に置く)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...backtest.mt5.spec import ORDER_TIME_SPECIFIED
from ...calendar import NS_HOUR, NS_MIN
from .. import STRATEGIES
from ..guards import size_lots
from ..indicators import atr, rsi, sma
from .base import MainStrategy


def server_day(t_ns: np.ndarray) -> np.ndarray:
    """サーバー時刻(ニューヨーク + 7 時間)の日付の通し番号。ロールオーバーが日の切り替わりになる。"""
    local = pd.DatetimeIndex(np.asarray(t_ns, dtype="datetime64[ns]"), tz="UTC").tz_convert("America/New_York")
    wall = local.tz_localize(None) + pd.Timedelta(hours=7)
    return wall.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)


@STRATEGIES.register("main_pullback_limit", version="1",
                     description="本命: 日足の押し目(200 日平均の向き + RSI(2))に、指値で待って入る(1 時間足で動かす)")
class MainPullbackLimit(MainStrategy):
    name = "main_pullback_limit"

    @dataclass
    class Params:
        timeframe: int = 60
        guard: dict = field(default_factory=dict)
        slow: int = 200
        rsi_n: int = 2
        enter: float = 10.0
        exit: float = 50.0
        max_days: int = 10
        atr_n: int = 20
        stop_atr: float = 3.0
        limit_atr: float = 0.5       # 0 = 成行
        tp_atr: float = 0.0          # 0 = 利確の指値を置かない
        decide_bars: int = 2         # 取引日が替わってから、この本数の足が確定したときに判断する
        expire_hours: float = 22.0
        vol_ratio_min: float = 0.0   # ATR(atr_n) / ATR(vol_long) がこれ以上の日だけ入る(0 = 見ない。E051)
        vol_long: int = 250
        notional: float = 0.0        # > 0 なら量を「有効証拠金 x これ」の建玉にする(ボラで量を変えない。E053)
        rolling: bool = False        # 日足未満の版で、区切りを 1 本ずつずらした全部の位相をまとめる(毎時判断。E052)
        signal_hours: float = 24.0   # 信号の足(時間)。24 未満は日足未満の版(E052)。max_days は足の本数

    def signals(self, bars, rng):    # MainStrategy の窓口は使わない(prepare と on_bar を自前で持つ)
        raise NotImplementedError

    def _phase(self, bars, off_ns: int, daily: bool) -> dict:
        """区切りを off_ns ずらした信号の足と、その指標・入りの向き・判断する足。"""
        p = self.p
        t = bars.t
        step = bars.timeframe * NS_MIN
        # 信号の足の区切り: 日足はサーバー時刻の取引日、それより短い足は UTC の時刻を signal_hours ごとに切る
        group = server_day if daily else (lambda x: (np.asarray(x) + off_ns) // int(p.signal_hours * NS_HOUR))
        day = group(t)
        uniq, inv = np.unique(day, return_inverse=True)
        starts = np.r_[0, np.flatnonzero(np.diff(inv)) + 1]
        ends = np.r_[starts[1:], len(t)]
        c = bars.close[ends - 1]
        d_high = np.maximum.reduceat(bars.high, starts)
        d_low = np.minimum.reduceat(bars.low, starts)
        # 信号の足の最後の足が欠けた版(先読み検査で切った系列)でも値が変わらないように、信号の足の値は
        # 「次の足が始まる時刻の区切り」より前のものだけを使う
        nxt = group(t + step)
        done = np.searchsorted(uniq, nxt, side="left") - 1               # 足 i の確定時点で、確定している最後の信号の足
        if daily:
            decide = (np.arange(len(t)) - starts[inv]) == p.decide_bars - 1
        else:
            decide = nxt != day                                           # 信号の足の最後の足が確定したとき
        ma = sma(c, p.slow)
        r = rsi(c, p.rsi_n)
        a = atr(d_high, d_low, c, p.atr_n)
        ok = np.isfinite(ma) & np.isfinite(r) & np.isfinite(a)
        if p.vol_ratio_min > 0:
            ratio = a / atr(d_high, d_low, c, p.vol_long)
            ok &= np.nan_to_num(ratio, nan=0.0) >= p.vol_ratio_min
        entry = np.where(ok & (c > ma) & (r < p.enter), 1, np.where(ok & (c < ma) & (r > 100 - p.enter), -1, 0))
        return {"c": c, "r": r, "a": a, "entry": entry, "done": done, "decide": decide}

    def prepare(self, bars, rng):
        p = self.p
        t = bars.t
        step = bars.timeframe * NS_MIN
        daily = p.signal_hours >= 24
        self.rolling = bool(p.rolling) and not daily
        n_ph = max(1, round(p.signal_hours * NS_HOUR / step)) if self.rolling else 1
        self.ph = [self._phase(bars, k * step, daily) for k in range(n_ph)]
        self.decide = np.any([q["decide"] for q in self.ph], axis=0)
        # 日足未満の版: ロールオーバーの前後(UTC 20〜22 時台)に始まる足では新規を出さない
        self.no_entry = (np.zeros(len(t), dtype=bool) if daily
                         else np.isin(((t + step) // NS_HOUR) % 24, (20, 21, 22)))
        self.expire_ns = int((p.expire_hours if daily else p.signal_hours) * NS_HOUR)
        self.warm_days = max(p.slow, p.atr_n, p.vol_long if p.vol_ratio_min > 0 else 0) + 1
        self.warmup = 0
        self.bars = bars
        self.age: dict[int, int] = {}
        self.signal_side: dict[int, int] = {}
        self.pos_phase: dict[int, int] = {}
        self.pending_signal = 0
        self.pending_phase = 0
        self.stats = {"limit": 0, "market": 0, "rejected": 0}

    def on_bar(self, i, ctx):
        if not self.decide[i]:
            return
        p = self.p
        now = [k for k, q in enumerate(self.ph) if q["decide"][i]]
        if not self.rolling:
            for o in ctx.orders(self.magic):             # 期限の切れ残り(念のため)
                ctx.order_delete(o.ticket)
        for pos in ctx.positions(self.magic):
            sig = self.signal_side.setdefault(pos.ticket, self.pending_signal or pos.side)
            k = self.pos_phase.setdefault(pos.ticket, self.pending_phase)
            if k not in now:                             # この建玉を建てた位相の足が確定したときだけ見る
                continue
            q = self.ph[k]
            d = int(q["done"][i])
            r = float(q["r"][d])
            age = self.age.get(pos.ticket, 0) + 1
            self.age[pos.ticket] = age
            if (r > p.exit if sig > 0 else r < 100 - p.exit) or age >= p.max_days:
                ctx.position_close(pos.ticket, reason="signal" if age < p.max_days else "time")
        if ctx.positions(self.magic) or not ctx.can_enter or self.no_entry[i]:
            return
        if self.rolling and ctx.orders(self.magic):      # 指値が待っている間は次の信号を見送る(1 度に 1 つ)
            return
        for k in now:
            q = self.ph[k]
            d = int(q["done"][i])
            if d < self.warm_days:
                continue
            signal = int(q["entry"][d])
            if signal == 0:
                continue
            want = signal
            if self.null_u is not None and self.null_u[i] >= 0.5:
                want = -signal
            a = float(q["a"][d])
            dist = p.stop_atr * a
            if p.notional > 0:
                sym = ctx.symbol()
                lots = ctx.normalize_volume(ctx.equity * p.notional / (ctx.ask * sym.contract_size))
            else:
                lots = size_lots(ctx, dist, self.g, want)
            if lots <= 0:
                return
            limit = float(q["c"][d]) - want * p.limit_atr * a
            quote = ctx.ask if want > 0 else ctx.bid
            self.pending_signal, self.pending_phase = signal, k
            if p.limit_atr <= 0 or (quote - limit) * want <= 0:      # 成行(気配がすでに指値より有利)
                tp = quote + want * p.tp_atr * a if p.tp_atr > 0 else None
                res = (ctx.buy if want > 0 else ctx.sell)(lots, sl=quote - want * dist, tp=tp)
                self.stats["market" if res.ok else "rejected"] += 1
                return
            limit = ctx.normalize_price(limit)
            tp = limit + want * p.tp_atr * a if p.tp_atr > 0 else None
            place = ctx.buy_limit if want > 0 else ctx.sell_limit
            res = place(lots, limit, sl=limit - want * dist, tp=tp, type_time=ORDER_TIME_SPECIFIED,
                        expiration=int(ctx.time + self.expire_ns))
            self.stats["limit" if res.ok else "rejected"] += 1
            return


__all__ = ["MainPullbackLimit"]
