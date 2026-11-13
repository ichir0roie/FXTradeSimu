"""E044: スワップ狙いの入りと出を EMA の向きだけで決める型(S044 main_carry_ema)を、B049 と口座の成績で比べる。

    python -m fxsim.research.carry_ema

利用者の案(2026-11-13): EMA だけを基準にし、1 回の取引は 5 日以上(週ごと)。ほかは推奨のまま(受け取る向きだけ・
金利差 2% 以上・キャリー・トゥ・リスク 0.4 以上・急変用の逆指値 ATR x 9・口座の床・1 ペア 2,000 通貨)。
E043 では、出を EMA10 < EMA30 にすると建玉比の年率が上がった(ペアごとに独立、口座の決まりなし)。口座の物差しで確かめる。
相場・口座・点数は E026 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ + 古いランダム 16 本、
10 万円、OANDA、約定は 2 時、swap_side_only)。
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
from fxsim.research.carry_harvest import NS_DAY
from fxsim.strategies import STRATEGIES

BASE = {"min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
GRID = [
    ("B049(10 日抜けで入り、5 日のチャネルで出る)", "main_carry_breakout", {**BASE, "n_in": 10, "n_out": 5}),
    ("10 日抜けで入り、EMA10/30 で出る", "main_carry_ema", {**BASE, "entry": "break"}),
    ("EMA10/30 で入り、出る(毎日判定)", "main_carry_ema", {**BASE}),
    ("EMA10/30・5 日は持つ", "main_carry_ema", {**BASE, "min_hold": 5}),
    ("EMA10/30・週に 1 回判定", "main_carry_ema", {**BASE, "signal_every": 5}),
    ("EMA5/20(毎日)", "main_carry_ema", {**BASE, "ema_fast": 5, "ema_slow": 20}),
    ("EMA5/20・週に 1 回判定", "main_carry_ema", {**BASE, "ema_fast": 5, "ema_slow": 20, "signal_every": 5}),
    ("EMA10/50(毎日)", "main_carry_ema", {**BASE, "ema_slow": 50}),
    ("EMA20/60(毎日)", "main_carry_ema", {**BASE, "ema_fast": 20, "ema_slow": 60}),
    ("EMA20/60・週に 1 回判定", "main_carry_ema", {**BASE, "ema_fast": 20, "ema_slow": 60, "signal_every": 5}),
    ("EMA50/150・週に 1 回判定(週足の 10/30)", "main_carry_ema",
     {**BASE, "ema_fast": 50, "ema_slow": 150, "signal_every": 5}),
]


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, strat, p) in enumerate(GRID):
        bt = build_backtester(md, STRATEGIES.create(strat, p), BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        days = [(x.exit_t - x.entry_t) / NS_DAY for x in bt.trades]
        pnl = sum(x.pnl for x in bt.trades)
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "per_year": len(bt.trades) / years,
                    "hold": float(np.median(days)) if days else 0.0,
                    "win": float(np.mean([x.pnl > 0 for x in bt.trades])) if bt.trades else 0.0,
                    "swap_share": float(sum(x.swap for x in bt.trades) / pnl) if pnl > 0 else float("nan")})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率・最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率・最大 DD(中央・最悪)",
            "取引 / 年・持つ日数の中央・勝率・スワップの割合(直近 / 古い実データ)",
            "B049 以上のシャープ(ランダム 24 本)", "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def med(x, f):
        return float(np.median([r[f] for r in x]))

    def rnd(x):
        return (f"{med(x, 'sharpe'):+.2f} / {med(x, 'cagr'):+.1%}・{med(x, 'dd'):.0%}・"
                f"{max(r['dd'] for r in x):.0%}")

    def side(r):
        sh = "-" if not np.isfinite(r["swap_share"]) else f"{r['swap_share']:.0%}"
        return f"{r['per_year']:.1f}・{r['hold']:.0f} 日・{r['win']:.0%}・{sh}"
    ref = {r["market"]: r["sharpe"] for r in rows if r["k"] == 0}
    for k, (label, _, _) in enumerate(GRID):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = next(r for r in rs if r["market"] == "real_recent")
        b = next(r for r in rs if r["market"] == "real_old")
        score = 0.5 * a["sharpe"] + 0.5 * med(ri, "sharpe")
        better = sum(r["sharpe"] >= ref[r["market"]] for r in ri + rq)
        bad = sum(bool(r["losscut"] or r["ruined"]) for r in rs)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | {side(a)} / {side(b)} | "
              f"{better}/{len(ri) + len(rq)} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
