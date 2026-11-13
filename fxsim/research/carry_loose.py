"""E037: 実証実験(デモ口座の常駐)のために、B049 の条件を緩めて取引を頻繁にする。どこまで緩めると口座が削られるか。

    python -m fxsim.research.carry_loose

振る値: 金利差の下限 min_carry(2%・1%・0.5%・0)x キャリー・トゥ・リスクの下限 c2r_min(0.4・0.2・0)
x 幅(2 週: 10 日抜け・5 日で出る / 1 週: 5 日抜け・2 日で出る)x 利確(なし / ATR x 0.5 で利確して入り直す、E034)。
量・証拠金の上限・急変用の逆指値・口座の床は B049 のまま(1 ペア 2,000 通貨、50%)。
相場・口座・点数は E026 と同じ。頻度は全期間と、直近 1 年(2025-11-10〜2026-11-10、今の金利の環境)で数える。
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.research.carry_breakout_opt import markets, metrics
from fxsim.research.carry_harvest import NS_WEEK, weekly
from fxsim.strategies import STRATEGIES

LAST_YEAR = np.datetime64("2025-11-10", "ns").astype(np.int64)
BASE = {"units": 2000, "margin_cap": 0.5}
SPANS = [("2 週", {"n_in": 10, "n_out": 5}), ("1 週", {"n_in": 5, "n_out": 2})]
TPS = [("利確なし", {}), ("ATR x 0.5 で利確・入り直す", {"tp_atr": 0.5, "reenter": True})]
GRID = [(f"金利差 {mc:.1%}・c2r {cr:g}・{sn}・{tn}", {**BASE, "min_carry": mc, "c2r_min": cr, **sp, **tp})
        for mc, cr, (sn, sp), (tn, tp) in itertools.product((0.02, 0.01, 0.005, 0.0), (0.4, 0.2, 0.0), SPANS, TPS)]


def last_year(bt) -> dict:
    tr = [x for x in bt.trades if x.exit_t >= LAST_YEAR]
    weeks = (int(bt.bars.t[-1]) - LAST_YEAR) / NS_WEEK
    wins = [x for x in tr if x.pnl > 0]
    win_weeks = {(x.exit_t - LAST_YEAR) // NS_WEEK for x in wins}
    return {"ly_trades_w": len(tr) / weeks, "ly_wins_w": len(wins) / weeks, "ly_win_weeks": len(win_weeks) / weeks,
            "ly_pnl": float(sum(x.pnl for x in tr)), "ly_pairs": len({x.pair for x in tr})}


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, p) in enumerate(GRID):
        bt = build_backtester(md, STRATEGIES.create("main_carry_breakout", p), BrokerConfig.preset("oanda"), cfg,
                              seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * 86_400_000_000_000)
        row = {"stage": stage, "market": name, "k": k, **metrics(bt), **weekly(bt), "per_year": len(bt.trades) / years}
        if name == "real_recent":
            row.update(last_year(bt))
        out.append(row)
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD", "ランダム 8 本 シャープ・最大 DD(中央・最悪)",
            "古い実データ シャープ / 年率 / 最大 DD", "古いランダム 16 本 シャープ(中央)・最大 DD(最悪)",
            "取引 / 週(直近 12 年 / 古い)", "利確のある週(直近 12 年 / 古い)",
            "直近 1 年: 取引 / 週・利確 / 週・利確のある週・損益(円)・ペア数", "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))
    for k, (label, _) in enumerate(GRID):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = next(r for r in rs if r["market"] == "real_recent")
        b = next(r for r in rs if r["market"] == "real_old")

        def med(x, f):
            return float(np.median([r[f] for r in x]))
        score = 0.5 * a["sharpe"] + 0.5 * med(ri, "sharpe")
        bad = sum(bool(r["losscut"] or r["ruined"]) for r in rs)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | "
              f"{med(ri, 'sharpe'):+.2f}・{med(ri, 'dd'):.0%}・{max(r['dd'] for r in ri):.0%} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | "
              f"{med(rq, 'sharpe'):+.2f}・{max(r['dd'] for r in rq):.0%} | "
              f"{a['per_year'] / 52.18:.2f} / {b['per_year'] / 52.18:.2f} | "
              f"{a['win_weeks']:.0%} / {b['win_weeks']:.0%} | "
              f"{a['ly_trades_w']:.2f}・{a['ly_wins_w']:.2f}・{a['ly_win_weeks']:.0%}・{a['ly_pnl']:+,.0f}・"
              f"{a['ly_pairs']} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
