"""E032: スワップ狙い(B049・B047)の「時期選び」の手柄を、持つ日数をそろえた帰無と比べて分ける。

    python -m fxsim.research.carry_timing_null

verify の方向でたらめ版(basket_direction_null)は、新規のたびに向きを乱数で決め直す。受け取る向きにしか入らない決まり
(swap_side_only)の下では、払う向きを引いた注文は弾かれ、次の足でまた引き直すので、本体とほぼ同じ時期に受け取る向きで入る。
これは時期をでたらめにした帰無ではない。そこで、本体の取引をペアごとの「持つ本数」の束として取り出し、同じペア・同じ本数・
受け取る向きのまま、時期だけをでたらめに置き直した帰無と比べる。

    N1  時期は、そのペアの受け取りがプラス(業者の取り分を引いて)の足ならどこでも
    N2  時期は、本体の下限(金利差 min_carry 以上・キャリー・トゥ・リスク c2r_min 以上)を満たす足だけ
        (下限は同じで、ブレイクアウトの時期選びだけを比べる)

量・急変用の逆指値・証拠金の上限は本体と同じ。証拠金が足りない足は、空くまで新規を遅らせる(持つ本数は変えない)。
口座の床(protect)は帰無には入れない。相場は verify と同じ 30 年(実データ 1 本 + 並べ替え 10 本、約定は 2 時)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from fxsim.strategies.basket import BasketStrategy

SPAN = ("1996-11-11", "2026-11-10")
BEHAVIORS = ["B049", "B047"]
BOOT_SEEDS = list(range(1, 11))
REAL_DRAWS = 30
BOOT_DRAWS = 10


class TimingNull(BasketStrategy):
    """plan[pair] = [(入りを決める足, 持つ本数), ...] のとおりに、受け取る向きで建てて決めた本数で手仕舞う。"""
    name = "timing_null"

    @dataclass
    class Params:
        timeframe: int = 1440

    def __init__(self, inner, plan: dict[str, list[tuple[int, int]]]):
        super().__init__()
        self.inner, self.plan = inner, plan

    def prepare(self, bars_by_pair, rng):
        self.inner.prepare(bars_by_pair, rng)
        self.warmup = 0
        self.queue = {p: sorted(v) for p, v in self.plan.items()}
        self.exit_at: dict[str, int] = {}
        self.entered = 0
        self.delayed = 0

    def on_bar(self, i, ctx):
        s, pb = self.inner, ctx._pb
        lev = pb.broker.leverage
        for pair in list(self.exit_at):
            if ctx.side(pair) == 0 and i > self.exit_at[pair][1] + 1:      # 急変用の逆指値・弾かれた注文
                del self.exit_at[pair]
            elif i >= self.exit_at[pair][0]:
                ctx.target(pair, 0)
                del self.exit_at[pair]
        used = sum(leg.notional(i) for leg in pb.legs.values()) / lev
        for pair, q in self.queue.items():
            if not q or q[0][0] > i or pair in self.exit_at or ctx.side(pair) != 0:
                continue
            side = int(s.side[pair][i])
            margin = s.p.units * s.c[pair][i] / lev
            if side == 0 or s.carry[pair][i] <= 0 or used + margin > s.p.margin_cap * ctx.equity:
                self.delayed += 1
                continue
            _, length = q.pop(0)
            a = s.atr[pair][i]
            sl = s.p.emergency_atr * a if s.p.emergency_atr > 0 and np.isfinite(a) else None
            ctx.target(pair, side, s.p.units, sl)
            self.exit_at[pair] = (i + length, i)
            used += margin
            self.entered += 1


def market(kind: str, seed: int):
    src = {"start": SPAN[0], "end": SPAN[1], "daily_fill_hour": 2}
    if kind == "real":
        return MARKETS.create("real_pairs", {"source": src}).generate(seed=1)
    return MARKETS.create("real_pairs_bootstrap", {"source": {**src, "block_days": 250}}).generate(seed=seed)


def run(md, strategy, behavior):
    b = load_cases().behaviors[behavior]
    cfg = EngineConfig(**{**b.engine})
    bt = build_backtester(md, strategy, BrokerConfig.preset(b.broker if isinstance(b.broker, str) else "oanda"), cfg,
                          seed=1)
    bt.run()
    return bt


def summary(bt) -> dict:
    _, eq = daily_equity(bt)
    r = np.diff(eq) / eq[:-1]
    return {"pnl": float(eq[-1] - eq[0]), "swap": float(sum(t.swap for t in bt.trades)),
            "trades": len(bt.trades), "bars": int(sum(t.bars_held for t in bt.trades)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(260)) if r.std() > 0 else 0.0}


def holdings(bt) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for t in bt.trades:
        out.setdefault(t.pair, []).append(max(1, int(t.bars_held)))
    return out


def place(lengths: dict[str, list[int]], ok: dict[str, np.ndarray], warmup: int, n: int,
          rng: np.random.Generator) -> tuple[dict[str, list[tuple[int, int]]], int]:
    """ペアごとに、同じ本数の取引を ok の足から乱数で重ならないように置く。置けなかった数も返す。"""
    plan, missed = {}, 0
    for pair, ls in lengths.items():
        cand = np.flatnonzero(ok[pair][:n - 2])
        cand = cand[cand >= warmup]
        busy = np.zeros(n + 1, dtype=bool)
        out = []
        for L in rng.permutation(np.asarray(ls)):
            for _ in range(300):
                if not len(cand):
                    break
                d = int(rng.choice(cand))
                if d + L + 1 < n and not busy[d:d + L + 2].any():
                    busy[d:d + L + 2] = True
                    out.append((d, int(L)))
                    break
            else:
                missed += 1
        plan[pair] = out
    return plan, missed


def job(spec):
    behavior, kind, seed, draws = spec
    b = load_cases().behaviors[behavior]
    params = b.strategy_params()
    md = market(kind, seed)
    main = run(md, STRATEGIES.create(b.strategy, params), behavior)
    lengths = holdings(main)
    out = {"behavior": behavior, "kind": kind, "seed": seed, "main": summary(main), "n1": [], "n2": []}
    if draws and draws[0] == 0:          # 方向でたらめ版(verify の帰無)と、本体と同じ取引の割合
        null = run(md, STRATEGIES.create(b.null_strategy(), {"inner": b.strategy, "inner_params": dict(b.params),
                                                              "timeframe": 1440}), behavior)
        key = {(t.pair, t.entry_t, t.exit_t, t.side) for t in main.trades}
        same = sum((t.pair, t.entry_t, t.exit_t, t.side) in key for t in null.trades)
        out["dir"] = {**summary(null), "same": same / max(1, len(null.trades))}
    probe = STRATEGIES.create(b.strategy, params)
    legs = md.meta["legs"]
    probe.prepare({p: leg.bars(1440) for p, leg in legs.items()}, np.random.default_rng(0))
    n = len(probe.t)
    p = probe.p
    ok1 = {pair: probe.carry[pair] > 0 for pair in probe.pairs}
    ok2 = {pair: (probe.carry[pair] >= p.min_carry) & np.nan_to_num(probe.c2r[pair] >= p.c2r_min)
           for pair in probe.pairs}
    for d in draws:
        for key, ok in (("n1", ok1), ("n2", ok2)):
            rng = np.random.default_rng([seed, d, 7 if key == "n1" else 8])
            plan, missed = place(lengths, ok, probe.warmup, n, rng)
            s = TimingNull(STRATEGIES.create(b.strategy, params), plan)
            bt = run(md, s, behavior)
            out[key].append({**summary(bt), "missed": missed, "entered": s.entered})
    return out


def specs():
    out = []
    for beh in BEHAVIORS:
        chunks = np.array_split(np.arange(REAL_DRAWS), 6)
        for k, c in enumerate(chunks):
            out.append((beh, "real", 1, ([0] if k == 0 else []) + [int(x) + 1 for x in c]))
        for s in BOOT_SEEDS:
            out.append((beh, "boot", s, [0] + list(range(1, BOOT_DRAWS + 1))))
    return out


def tstat(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(job, specs()))
    print("## 実データ 30 年(1996-11〜2026-11)\n")
    print("| 動作ケース | 本体 損益 / スワップ / 取引 / 持った本数 | "
          "方向でたらめ版 損益(本体と同じ取引の割合) | N1 損益 平均 [5%・95%](本体以上の割合) | "
          "N2 損益 平均 [5%・95%](本体以上の割合) | 置けなかった取引(N1 / N2) |")
    print("|---|---:|---:|---:|---:|---:|")
    for beh in BEHAVIORS:
        rs = [r for r in rows if r["behavior"] == beh and r["kind"] == "real"]
        m = rs[0]["main"]
        d = next(r["dir"] for r in rs if "dir" in r)
        cells = []
        for key in ("n1", "n2"):
            v = np.array([x["pnl"] for r in rs for x in r[key]])
            cells.append(f"{v.mean():+,.0f} [{np.percentile(v, 5):+,.0f}・{np.percentile(v, 95):+,.0f}]"
                         f"({np.mean(v >= m['pnl']):.0%})")
        miss = [np.mean([x["missed"] for r in rs for x in r[k]]) for k in ("n1", "n2")]
        print(f"| {beh} | {m['pnl']:+,.0f} / {m['swap']:+,.0f} / {m['trades']} / {m['bars']} | "
              f"{d['pnl']:+,.0f}({d['same']:.0%}) | {cells[0]} | {cells[1]} | {miss[0]:.1f} / {miss[1]:.1f} |")
    print("\n## 並べ替え 10 本(250 日ブロック、金利差も日付と一緒)\n")
    print("| 動作ケース | 本体 損益 平均 | 方向でたらめ版 平均(同じ取引の割合)・差の t | N1 平均・差の平均 / t | "
          "N2 平均・差の平均 / t | スワップ 本体 / N1 / N2 | シャープ 本体 / N1 / N2(中央) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for beh in BEHAVIORS:
        rs = [r for r in rows if r["behavior"] == beh and r["kind"] == "boot"]
        m = np.array([r["main"]["pnl"] for r in rs])
        dv = np.array([r["dir"]["pnl"] for r in rs])
        same = np.mean([r["dir"]["same"] for r in rs])
        n1 = np.array([np.mean([x["pnl"] for x in r["n1"]]) for r in rs])
        n2 = np.array([np.mean([x["pnl"] for x in r["n2"]]) for r in rs])
        sw = [np.mean([r["main"]["swap"] for r in rs]), np.mean([x["swap"] for r in rs for x in r["n1"]]),
              np.mean([x["swap"] for r in rs for x in r["n2"]])]
        sh = [np.median([r["main"]["sharpe"] for r in rs]), np.median([x["sharpe"] for r in rs for x in r["n1"]]),
              np.median([x["sharpe"] for r in rs for x in r["n2"]])]
        print(f"| {beh} | {m.mean():+,.0f} | {dv.mean():+,.0f}({same:.0%})・{tstat(m - dv):.2f} | "
              f"{n1.mean():+,.0f}・{(m - n1).mean():+,.0f} / {tstat(m - n1):.2f} | "
              f"{n2.mean():+,.0f}・{(m - n2).mean():+,.0f} / {tstat(m - n2):.2f} | "
              f"{sw[0]:+,.0f} / {sw[1]:+,.0f} / {sw[2]:+,.0f} | {sh[0]:+.2f} / {sh[1]:+.2f} / {sh[2]:+.2f} |")


if __name__ == "__main__":
    main()
