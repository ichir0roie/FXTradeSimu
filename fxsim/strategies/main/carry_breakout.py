"""S040 main_carry_breakout: 本命の 3 本目。主要 10 ペアを 1 つの口座で持つ、スワップ狙い + ブレイクアウト。

E012〜E025 のまとめ。
    ベース      スワップを受け取る向きだけ。業者の取り分(recv_markup)を引いた受け取りが年 min_carry 以上のペアが候補
                (E018・E022: 払う向きに入らないと日足は全部良くなる。中身はキャリー)
    選び方      値の振れに比べて金利差が大きいペアだけ
                (キャリー・トゥ・リスク = 受け取りの年率 / vol_n 日ボラの年率 ≥ c2r_min)。
                候補が多いときはキャリー・トゥ・リスクの高い順(E024: 同じ稼ぎを小さいドローダウンで取れる)
    シグナル    終値が直近 n_in 日の高値(売りは安値)を抜けたら入る
                (E023・E025: ブレイクアウトは落ちていく相場に入らない保険)
    手仕舞い    終値が反対側の n_out 日の安値(売りは高値)を割る、金利差が下限を割る・向きが変わる、
                または値動き + スワップの合計が建玉の take_profit 以上
                (0 なら使わない。E024 では利確の工夫は効かなかった)
    共通の対策  月初の有効証拠金 x protect を割ったら全部手仕舞い、その月は新規を出さない(口座の床)。
                必要証拠金の合計を有効証拠金 x margin_cap までに抑える。
                サーバーには急変用の逆指値を ATR x emergency_atr に置く(E019: ATR x 9 は誤作動なし)。
                スプレッドが典型の max_spread_mult 倍を超えたら新規・信号の手仕舞いを出さない。
                全部の動作ケースと同じく swap_side_only(エンジンの設定)
    量          1 ペア units 通貨(OANDA の最小は 1,000)。10 万円の口座では 1,000 通貨で 1 ペア 1〜2 倍(E025)

相場は real_pairs(R013)/ real_pairs_bootstrap(R014)。金利差は、相場が足ごとの金利差を持てばそれ(並べ替えた相場)、
無ければペアの短期金利の差(8 通貨の月次)。複数ペアを 1 つの口座で回す(PortfolioBacktester)。
MT5 の常駐は 1 銘柄ずつなので未対応。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...data import load_rates
from ...data.rates import MULTI_RATES
from .. import STRATEGIES
from ..basket import BasketStrategy
from ..guards import EquityFloor, typical_spread
from ..indicators import atr, highest, lowest


@STRATEGIES.register("main_carry_breakout", version="1",
                     description="本命: 10 ペアの口座でスワップを受け取る向きだけ、ブレイクアウトで入り"
                                 "反対側のチャネルで出る")
class MainCarryBreakout(BasketStrategy):
    name = "main_carry_breakout"
    tier: ClassVar[str] = "main"
    pair_market: ClassVar[str] = "real_pairs"      # pair-eval で使う相場(既定の basket_* ではなく)

    @dataclass
    class Params:
        timeframe: int = 1440
        n_in: int = 55
        n_out: int = 20                 # 0 = 反対側のチャネルでは出ない(金利差が無くなるまで持つ)
        min_carry: float = 0.01
        recv_markup: float = 0.0044
        c2r_min: float = 0.2            # 0 = 見ない
        vol_n: int = 60
        units: float = 1000.0
        margin_cap: float = 0.5
        atr_n: int = 20
        emergency_atr: float = 9.0      # 0 = 置かない
        take_profit: float = 0.0        # 0 = 使わない
        protect: float = 0.9            # 0 = 床を置かない
        max_spread_mult: float = 3.0    # 0 = 見ない
        spread_window: int = 20

    def prepare(self, bars_by_pair, rng):
        p = self.p
        self.pairs = list(bars_by_pair)
        self.t = bars_by_pair[self.pairs[0]].t
        rates = load_rates(MULTI_RATES)
        self.c, self.side, self.carry, self.c2r = {}, {}, {}, {}
        self.hi_in, self.lo_in, self.hi_out, self.lo_out, self.atr = {}, {}, {}, {}, {}
        self.spread, self.typical = {}, {}
        for pair, b in bars_by_pair.items():
            diff = b.rate_diff if b.rate_diff is not None else rates.pair_differential(pair, b.t)
            self.c[pair] = b.close
            self.side[pair] = np.sign(diff).astype(int)
            self.carry[pair] = np.abs(diff) - p.recv_markup
            r = np.diff(np.log(b.close), prepend=np.log(b.close[0]))
            vol = np.full(len(r), np.nan)
            for i in range(p.vol_n, len(r)):
                vol[i] = r[i - p.vol_n + 1:i + 1].std() * np.sqrt(260)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.c2r[pair] = self.carry[pair] / vol
            self.hi_in[pair] = np.r_[np.nan, highest(b.high, p.n_in)[:-1]]
            self.lo_in[pair] = np.r_[np.nan, lowest(b.low, p.n_in)[:-1]]
            n_out = p.n_out or p.n_in
            self.hi_out[pair] = np.r_[np.nan, highest(b.high, n_out)[:-1]]
            self.lo_out[pair] = np.r_[np.nan, lowest(b.low, n_out)[:-1]]
            self.atr[pair] = atr(b.high, b.low, b.close, p.atr_n)
            self.spread[pair] = b.spread_mean
            self.typical[pair] = typical_spread(b.spread_mean, p.spread_window)
        self.warmup = max(p.vol_n, p.n_in, p.n_out, p.atr_n) + 1
        self.floor = EquityFloor(p.protect)
        self.blocked = {"floor": 0, "margin": 0, "spread": 0, "spread_exit": 0, "c2r": 0}
        self.held: list[int] = []

    def _wide(self, pair: str, i: int) -> bool:
        m = self.p.max_spread_mult
        ty = self.typical[pair][i]
        return m > 0 and np.isfinite(ty) and ty > 0 and self.spread[pair][i] > m * ty

    def _want(self, pair: str, i: int) -> int:
        return int(self.side[pair][i]) if self.carry[pair][i] >= self.p.min_carry else 0

    def on_bar(self, i, ctx):
        p, pb = self.p, ctx._pb
        eq = ctx.equity
        # 口座の床
        if p.protect > 0:
            self.floor.update(int(self.t[i]), eq)
            if self.floor.check(eq):
                ctx.close_all()
                self.blocked["floor"] += 1
                return
            if self.floor.halted:
                return
        lev = pb.broker.leverage
        used = 0.0
        held = 0
        # 手仕舞い
        for pair in self.pairs:
            cur = ctx.side(pair)
            if cur == 0:
                continue
            c = self.c[pair][i]
            out = self._want(pair, i) != cur
            if p.n_out > 0:
                out |= (c < self.lo_out[pair][i]) if cur > 0 else (c > self.hi_out[pair][i])
            pos = pb.legs[pair].pos
            if p.take_profit > 0 and pos is not None:
                total = (c - pos.entry_price) * pos.side * pos.units + pos.swap
                out |= total >= p.take_profit * pos.units * pos.entry_price
            if out and self._wide(pair, i):
                self.blocked["spread_exit"] += 1
                out = False
            if out:
                ctx.target(pair, 0)
                continue
            held += 1
            if pos is not None:
                used += pos.units * c / lev
        # 新規(キャリー・トゥ・リスクの高い順)
        cands = []
        for pair in self.pairs:
            if ctx.side(pair) != 0:
                continue
            side = self._want(pair, i)
            if side == 0:
                continue
            c2r = self.c2r[pair][i]
            if p.c2r_min > 0 and not (np.isfinite(c2r) and c2r >= p.c2r_min):
                self.blocked["c2r"] += 1
                continue
            c = self.c[pair][i]
            broke = c > self.hi_in[pair][i] if side > 0 else c < self.lo_in[pair][i]
            if not broke:
                continue
            if self._wide(pair, i):
                self.blocked["spread"] += 1
                continue
            cands.append((-(c2r if np.isfinite(c2r) else -1e9), pair, side))
        for _, pair, side in sorted(cands):
            c = self.c[pair][i]
            margin = p.units * c / lev
            if used + margin > p.margin_cap * eq:
                self.blocked["margin"] += 1
                continue
            a = self.atr[pair][i]
            sl = p.emergency_atr * a if p.emergency_atr > 0 and np.isfinite(a) else None
            ctx.target(pair, side, p.units, sl)
            used += margin
            held += 1
        self.held.append(held)


__all__ = ["MainCarryBreakout"]
