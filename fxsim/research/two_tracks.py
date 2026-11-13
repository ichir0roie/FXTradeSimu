"""E047: スワップ狙いを 2 本に分ける。A = 順位の上位を常に持つ(S046)、B = 受け取る向きの信号で入り数日で出る(S047)。

    python -m fxsim.research.two_tracks

利用者の方針(2026-11-13): 分析手法の優位性は 3 日ほどしか無く、今の基本戦略はスワップと長期の持ち方が前提。
分けて考える。
    A  金利差の上位 5 ペアを常に持つ。順位から外れたら、損益がプラスか損切りで閉じる。長期的に受け取る向きに動いていない
       ペアには入らない
    B  受け取る向きのまま、信号が点いたら入り、3 日で閉じる
相場・点数は E026 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ + 古いランダム 16 本、OANDA、
約定は 2 時、swap_side_only)。口座は 10 万円。A は 1 ペア 1,000 通貨(OANDA の最小。5 ペアで建玉は口座の 6〜7 倍)。
"""
from __future__ import annotations

import sys
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

B049 = ("main_carry_breakout", {"n_in": 10, "n_out": 5, "min_carry": 0.02, "c2r_min": 0.4, "units": 2000,
                                "margin_cap": 0.5})
HOLD = {"units": 1000, "margin_cap": 0.5}
SHORT = {"units": 2000, "margin_cap": 0.5}
GRID_A = [
    ("B049(基準)", *B049),
    ("A 上位 5・長期 250 日・外れたらプラスか損切り ATR x 3(利用者の案)", "main_carry_hold", {**HOLD}),
    ("A 同じ・外れたらすぐ閉じる", "main_carry_hold", {**HOLD, "exit": "now"}),
    ("A 同じ・長期の条件なし", "main_carry_hold", {**HOLD, "lt_n": 0}),
    ("A 同じ・長期 125 日", "main_carry_hold", {**HOLD, "lt_n": 125}),
    ("A 同じ・長期 60 日", "main_carry_hold", {**HOLD, "lt_n": 60}),
    ("A 同じ・損切り ATR x 6", "main_carry_hold", {**HOLD, "stop_atr": 6.0}),
    ("A 同じ・上位 3", "main_carry_hold", {**HOLD, "top_n": 3}),
    ("A 同じ・順位をキャリー・トゥ・リスクで", "main_carry_hold", {**HOLD, "rank_by": "c2r"}),
    ("A 同じ・金利差 2% 以上のペアだけ", "main_carry_hold", {**HOLD, "min_carry": 0.02}),
    ("A 同じ・金利差 2% 以上・c2r 順・上位 3", "main_carry_hold",
     {**HOLD, "min_carry": 0.02, "rank_by": "c2r", "top_n": 3}),
]
GRID_B = [
    ("B049(基準)", *B049),
    ("B 10 日抜け・3 日・点き直したら延ばす(利用者の案)", "main_carry_short", {**SHORT}),
    ("B 同じ・延ばさない", "main_carry_short", {**SHORT, "extend": False}),
    ("B 同じ・1 日", "main_carry_short", {**SHORT, "hold": 1}),
    ("B 同じ・2 日", "main_carry_short", {**SHORT, "hold": 2}),
    ("B 同じ・5 日", "main_carry_short", {**SHORT, "hold": 5}),
    ("B 同じ・10 日", "main_carry_short", {**SHORT, "hold": 10}),
    ("B MACD の交差・3 日", "main_carry_short", {**SHORT, "signal": "macd"}),
    ("B 抜けか MACD・3 日", "main_carry_short", {**SHORT, "signal": "either"}),
    ("B 10 日抜け・3 日・下限をゆるく(金利差 1%・c2r 0.2)", "main_carry_short",
     {**SHORT, "min_carry": 0.01, "c2r_min": 0.2}),
    ("B 10 日抜け・3 日・5,000 通貨", "main_carry_short", {**SHORT, "units": 5000}),
]
GRIDS = {"A": GRID_A, "B": GRID_B}


def job(spec):
    which, (stage, name, model, params) = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, strat, p) in enumerate(GRIDS[which]):
        bt = build_backtester(md, STRATEGIES.create(strat, p), BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        swap = sum(x.swap for x in bt.trades)
        pnl = sum(x.pnl for x in bt.trades)
        days = [(x.exit_t - x.entry_t) / NS_DAY for x in bt.trades]
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "per_year": len(bt.trades) / years,
                    "swap": swap / years, "move": (pnl - swap) / years,
                    "hold": float(np.median(days)) if days else 0.0,
                    "win": float(np.mean([x.pnl > 0 for x in bt.trades])) if bt.trades else 0.0})
    return out


def table(which: str, rows: list[dict]) -> None:
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率・最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率・最大 DD(中央・最悪)",
            "値動き 円 / 年: 直近・古い実データ・ランダム 24 本の平均(t)", "スワップ 円 / 年(同じ並び)",
            "取引 / 年・持つ日数・勝率(直近)", "B049 以上のシャープ(ランダム 24 本)", "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def med(x, f):
        return float(np.median([r[f] for r in x]))

    def rnd(x):
        return (f"{med(x, 'sharpe'):+.2f} / {med(x, 'cagr'):+.1%}・{med(x, 'dd'):.0%}・"
                f"{max(r['dd'] for r in x):.0%}")

    def tstat(v):
        v = np.asarray(v, float)
        return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else 0.0
    ref = {r["market"]: r["sharpe"] for r in rows if r["k"] == 0}
    for k, (label, _, _) in enumerate(GRIDS[which]):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = next(r for r in rs if r["market"] == "real_recent")
        b = next(r for r in rs if r["market"] == "real_old")
        score = 0.5 * a["sharpe"] + 0.5 * med(ri, "sharpe")
        rr = ri + rq
        mv, sw = [r["move"] for r in rr], [r["swap"] for r in rr]
        better = sum(r["sharpe"] >= ref[r["market"]] for r in rr)
        bad = sum(bool(r["losscut"] or r["ruined"]) for r in rs)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{a['move']:+,.0f}・{b['move']:+,.0f}・{np.mean(mv):+,.0f}({tstat(mv):+.1f}) | "
              f"{a['swap']:+,.0f}・{b['swap']:+,.0f}・{np.mean(sw):+,.0f} | "
              f"{a['per_year']:.0f}・{a['hold']:.0f} 日・{a['win']:.0%} | {better}/{len(rr)} | {bad}/{len(rs)} |")


def main() -> None:
    for which in (sys.argv[1:] or ["A", "B"]):
        with ProcessPoolExecutor(6) as ex:
            rows = [r for part in ex.map(job, [(which, m) for m in markets()]) for r in part]
        print(f"\n## {which}\n")
        table(which, rows)


if __name__ == "__main__":
    main()
