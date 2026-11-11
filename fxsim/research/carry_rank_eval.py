"""E028: 本命 3 本目の「順位で選び、上から 4:3:2:1 の割合で張る」版(S041 main_carry_rank)を、B047 と同じ相場で比べる。

    python -m fxsim.research.carry_rank_eval

相場は E026 と同じ(選ぶ段: 直近の実データ 2014-11〜2026-11 + ランダム 8 本、確かめる段: 古い実データ 1996〜2014-10 +
古いランダム 16 本)。口座は 10 万円、OANDA、約定は 2 時、swap_side_only。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.research.carry_breakout_opt import markets, metrics
from fxsim.strategies import STRATEGIES

B047 = {"n_in": 20, "n_out": 10, "min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
RANK = {"n_in": 20, "n_out": 10, "min_carry": 0.01, "margin_cap": 0.5}
VARIANTS = [
    ("B047(全ペア 2,000 通貨、金利差 2%・c2r 0.4)", "main_carry_breakout", B047),
    ("上位 4、4:3:2:1", "main_carry_rank", {**RANK, "weights": [4, 3, 2, 1]}),
    ("上位 5、4:3:2:1:1", "main_carry_rank", {**RANK, "weights": [4, 3, 2, 1, 1]}),
    ("上位 5、5:4:3:2:1", "main_carry_rank", {**RANK, "weights": [5, 4, 3, 2, 1]}),
    ("上位 5、2:2:2:2:2(割合の効き目を見る)", "main_carry_rank", {**RANK, "weights": [2, 2, 2, 2, 2]}),
    ("上位 4、4:3:2:1、金利差 2%", "main_carry_rank", {**RANK, "weights": [4, 3, 2, 1], "min_carry": 0.02}),
    ("上位 4、4:3:2:1、B047 の下限(金利差 2%・c2r 0.4)", "main_carry_rank",
     {**RANK, "weights": [4, 3, 2, 1], "min_carry": 0.02, "c2r_min": 0.4}),
    ("上位 4、2,000:1,500:1,000:1,000 通貨、B047 の下限", "main_carry_rank",
     {**RANK, "weights": [4, 3, 2, 2], "unit": 500, "min_carry": 0.02, "c2r_min": 0.4}),
]


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, strat, p) in enumerate(VARIANTS):
        s = STRATEGIES.create(strat, p)
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        top = getattr(s, "rank_hist", None)
        held = float(np.mean(s.held)) if s.held else 0.0
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * 86_400_000_000_000)
        days = [(t.exit_t - t.entry_t) / 86_400_000_000_000 for t in bt.trades]
        churn = {"per_year": len(bt.trades) / years, "hold_med": float(np.median(days)) if days else 0.0,
                 "short": float(np.mean([d < 5 for d in days])) if days else 0.0}
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "held": held, **churn,
                    "top": dict(top) if top else None})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪) / プラス", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率 / 最大 DD(中央・最悪) / プラス", "ロスカット・破綻(25 本)",
            "平均の建玉数(直近の実データ)", "取引 / 年・持つ日数の中央・5 日未満の割合(直近 / 古い実データ)"]
    print("| " + " | ".join(cols) + " |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, (label, _, _) in enumerate(VARIANTS):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = [r for r in rs if r["market"] == "real_recent"][0]
        b = [r for r in rs if r["market"] == "real_old"][0]

        def rnd(x):
            med = {f: float(np.median([r[f] for r in x])) for f in ("sharpe", "cagr", "dd")}
            return (f"{med['sharpe']:+.2f} / {med['cagr']:+.1%} / "
                    f"{med['dd']:.0%}・{max(r['dd'] for r in x):.0%} / "
                    f"{np.mean([r['cagr'] > 0 for r in x]):.0%}")
        bad = sum(r["losscut"] > 0 or r["ruined"] for r in rs)
        print(f"| {label} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | {bad}/{len(rs)} | "
              f"{a['held']:.1f} | {a['per_year']:.1f}・{a['hold_med']:.0f} 日・{a['short']:.0%} / "
              f"{b['per_year']:.1f}・{b['hold_med']:.0f} 日・{b['short']:.0%} |")
    for k in (1,):
        a = [r for r in rows if r["k"] == k and r["market"] == "real_recent"][0]
        b = [r for r in rows if r["k"] == k and r["market"] == "real_old"][0]
        print(f"\n1 位だった日数(直近の実データ): {a['top']}")
        print(f"1 位だった日数(古い実データ): {b['top']}")


if __name__ == "__main__":
    main()
