"""E025: 10 万円・25 倍の 1 つの口座で、キャリー + ブレイクアウトを主要 10 ペアに回す。

    python -m fxsim.research.carry_account

相場は real_pairs(R013、日足 10 ペア、日付をそろえる、約定は 2 時)、ブローカーは oanda(25 倍、証拠金維持率 100% で
ロスカット、ゼロカットなし)、全部 swap_side_only。口座は 10 万円から、6 年 x 5 期間を別々に回す。
入る候補は「スワップを受け取る向きで、業者の取り分を引いた受け取りが年 1% 以上」のペア。

入り方と手仕舞い(回す前に決めた 4 つ):
    A 持ち続ける           いつでも入る。金利差が無くなる(向きが変わる)まで持つ
    B ブレイク 20 / 10     終値が直近 20 日の高値(売りは安値)を抜けたら入る。反対側の 10 日を割ったら手仕舞う
    C ブレイク 55 / 20     55 日で入り、反対側の 20 日で手仕舞う
    D ブレイクで入り持つ   20 日抜けで入り、金利差が無くなるか、建玉の損失(スワップ込み)が −10% で手仕舞う
B〜D には、同じ手仕舞いで「同じ頻度で、でたらめな日に入る」版(乱数 5 本)を対にする。
ブレイクの手柄が入る時期にあるかを見る。

量: 1 ペア units 通貨(既定 1,000 = OANDA の最小)。建てた後の必要証拠金の合計が「有効証拠金 x cap」を超えない範囲で、
候補をキャリー・トゥ・リスク(受け取りの年率 / 60 日ボラの年率)の高い順に建てる。
cap 1.0 なら証拠金を使い切る(維持率 100%)。
ロスカットは足の終値で口座全体を見る(実際は足の中で掛かるので、ここの数字は楽観的)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.data import load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.pair_eval import RANGES
from fxsim.strategies.indicators import highest, lowest

RECV = 0.0044
INITIAL = 100_000.0


@dataclass(frozen=True)
class Plan:
    name: str
    entry: str = "any"           # any | breakout | random
    n_in: int = 20
    n_out: int | None = None     # 反対側のチャネルで手仕舞う(None = 金利差が無くなるまで)
    stop: float | None = None    # 建玉の損失(スワップ込み)/ 建玉の金額 がこれ以下で手仕舞う
    units: float = 1000.0
    cap: float = 0.5             # 必要証拠金の合計 / 有効証拠金 の上限


class CarryAccount:
    multi = True
    timeframe = 1440

    def __init__(self, plan: Plan, seed: int = 0):
        self.plan, self.seed = plan, seed

    def prepare(self, bars_by_pair, rng):
        self.pairs = list(bars_by_pair)
        b0 = bars_by_pair[self.pairs[0]]
        rt = load_rates(MULTI_RATES)
        self.c = {p: b.close for p, b in bars_by_pair.items()}
        self.side, self.carry, self.c2r, self.up, self.dn = {}, {}, {}, {}, {}
        self.xup, self.xdn, self.prob = {}, {}, {}
        n_in, n_out = self.plan.n_in, self.plan.n_out or self.plan.n_in
        for p, b in bars_by_pair.items():
            diff = rt.pair_differential(p, b.t)
            self.side[p] = np.sign(diff).astype(int)
            self.carry[p] = np.abs(diff) - RECV
            r = np.diff(np.log(b.close), prepend=np.log(b.close[0]))
            vol = np.full(len(r), np.nan)
            for i in range(60, len(r)):
                vol[i] = r[i - 59:i + 1].std() * np.sqrt(260)
            self.c2r[p] = self.carry[p] / vol
            hi = np.r_[np.nan, highest(b.high, n_in)[:-1]]
            lo = np.r_[np.nan, lowest(b.low, n_in)[:-1]]
            self.up[p] = b.close > hi
            self.dn[p] = b.close < lo
            self.xup[p] = np.r_[np.nan, highest(b.high, n_out)[:-1]]
            self.xdn[p] = np.r_[np.nan, lowest(b.low, n_out)[:-1]]
            ok = self.carry[p] >= 0.01
            sig = np.where(self.side[p] > 0, self.up[p], self.dn[p]) & ok
            self.prob[p] = float(sig.sum() / max(ok.sum(), 1))       # でたらめな日に入る版の頻度(ブレイクと同じ)
        self.rng = np.random.default_rng([self.seed, 2027])
        self.t = b0.t
        self.warmup = max(60, n_in, n_out) + 1
        self.blocked_month: dict[str, int] = {}
        self.held, self.lev = [], []

    def _signal(self, p: str, i: int, side: int) -> bool:
        e = self.plan.entry
        if e == "any":
            return True
        if e == "breakout":
            return bool(self.up[p][i] if side > 0 else self.dn[p][i])
        return bool(self.rng.random() < self.prob[p])

    def on_bar(self, i, ctx):
        pb = ctx._pb
        lev = pb.broker.leverage
        month = int(np.datetime64(int(self.t[i]), "ns").astype("datetime64[M]").astype(np.int64))
        used = 0.0
        open_pairs = []
        for p in self.pairs:
            leg = pb.legs[p]
            pos = leg.pos
            if pos is None:
                continue
            c = self.c[p][i]
            want = int(self.side[p][i]) if self.carry[p][i] >= 0.01 else 0
            out = pos.side != want
            if self.plan.n_out is not None:
                out |= (c < self.xdn[p][i]) if pos.side > 0 else (c > self.xup[p][i])
            if self.plan.stop is not None:
                pnl = (c - pos.entry_price) * pos.side * pos.units + pos.swap
                if pnl <= self.plan.stop * pos.units * pos.entry_price:
                    out = True
                    self.blocked_month[p] = month
            if out:
                ctx.target(p, 0)
            else:
                used += pos.units * c / lev
                open_pairs.append(p)
        eq = ctx.equity
        cands = []
        for p in self.pairs:
            if p in open_pairs or pb.legs[p].pos is not None:
                continue
            side = int(self.side[p][i])
            if self.carry[p][i] < 0.01 or side == 0 or self.blocked_month.get(p) == month:
                continue
            if not self._signal(p, i, side):
                continue
            cands.append((-(self.c2r[p][i] if np.isfinite(self.c2r[p][i]) else -1e9), p, side))
        for _, p, side in sorted(cands):
            margin = self.plan.units * self.c[p][i] / lev
            if used + margin > self.plan.cap * eq:
                continue
            ctx.target(p, side, self.plan.units)
            used += margin
            open_pairs.append(p)
        self.held.append(len(open_pairs))
        self.lev.append(used * lev / eq if eq > 0 else np.nan)


PLANS = [Plan("A 持ち続ける", "any"), Plan("B ブレイク 20 / 10", "breakout", 20, 10),
         Plan("C ブレイク 55 / 20", "breakout", 55, 20),
         Plan("D ブレイクで入り持つ(−10%)", "breakout", 20, None, -0.10)]
SIZES = [(1000.0, 0.5), (1000.0, 1.0), (1000.0, 0.25), (2000.0, 0.5)]


def run_one(start: str, end: str, plan: Plan, seed: int = 0) -> dict:
    md = MARKETS.create("real_pairs", {"source": {"start": start, "end": end, "daily_fill_hour": 2}}).generate(seed=1)
    s = CarryAccount(plan, seed)
    cfg = EngineConfig(initial_balance=INITIAL, swap_side_only=True, min_units=1000.0)
    bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
    bt.run()
    _, eq = daily_equity(bt)
    years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * 86_400_000_000_000)
    peak = np.maximum.accumulate(eq)
    final = float(eq[-1])
    return {"start": start, "plan": plan, "seed": seed, "final": final,
            "cagr": (max(final, 1.0) / INITIAL) ** (1 / years) - 1, "dd": float(((peak - eq) / peak).max()),
            "losscut": int(bt.stats.get("losscut", 0)), "ruined": bool(bt.ruined),
            "held": float(np.nanmean(s.held)) if s.held else 0.0, "lev": float(np.nanmean(s.lev)) if s.lev else 0.0,
            "lev_max": float(np.nanmax(s.lev)) if s.lev else 0.0, "trades": len(bt.trades),
            "swap": float(sum(t.swap for t in bt.trades))}


def _job(arg):
    return run_one(*arg)


def main() -> None:
    windows = RANGES["D1"].windows
    jobs = []
    for units, cap in SIZES:
        for plan in PLANS:
            p = Plan(plan.name, plan.entry, plan.n_in, plan.n_out, plan.stop, units, cap)
            jobs += [(s, e, p, 0) for s, e in windows]
            if plan.entry == "breakout" and (units, cap) == SIZES[0]:
                q = Plan(plan.name + " のでたらめ版", "random", plan.n_in, plan.n_out, plan.stop, units, cap)
                jobs += [(s, e, q, seed) for s, e in windows for seed in range(1, 6)]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(_job, jobs))
    print("| 量 / 証拠金の上限 | 入り方 | 年率(5 期間の平均) | 6 年後(平均 / 最小、万円) | 最大 DD(平均 / 最悪) "
          "| ロスカット | "
          "破綻 | 持つペア(平均) | 実質レバレッジ(平均 / 最大) | 前半 2 期間の年率 | 後半 3 期間の年率 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    keys = list(dict.fromkeys((r["plan"].units, r["plan"].cap, r["plan"].name) for r in rows))
    first = {w[0] for w in windows[:2]}
    for units, cap, name in keys:
        rs = [r for r in rows if (r["plan"].units, r["plan"].cap, r["plan"].name) == (units, cap, name)]
        cagr = np.array([r["cagr"] for r in rs])
        fin = np.array([r["final"] for r in rs]) / 1e4
        dd = np.array([r["dd"] for r in rs])
        c1 = np.mean([r["cagr"] for r in rs if r["start"] in first])
        c2 = np.mean([r["cagr"] for r in rs if r["start"] not in first])
        print(f"| {units:,.0f} 通貨 / {cap:.0%} | {name} | {cagr.mean():+.1%} | {fin.mean():.1f} / {fin.min():.1f} | "
              f"{dd.mean():.0%} / {dd.max():.0%} | {sum(r['losscut'] > 0 for r in rs)}/{len(rs)} | "
              f"{sum(r['ruined'] for r in rs)}/{len(rs)} | {np.mean([r['held'] for r in rs]):.1f} | "
              f"{np.mean([r['lev'] for r in rs]):.1f} / {max(r['lev_max'] for r in rs):.1f} | {c1:+.1%} | {c2:+.1%} |")


if __name__ == "__main__":
    main()
