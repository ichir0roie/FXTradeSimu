"""E041: スワップ狙い(B049・B054)の利確を、終値の判定からサーバーの指値に替える。どの幅なら成績を崩さずに置けるか。

    python -m fxsim.research.carry_tp_order

利用者の判断(2026-11-12): デモ口座の常駐でも、建玉には利確の指値を置く(リスク管理の基本として)。
B049 は利確が無い(反対側の 5 日のチャネルを終値で割ったら出る)。B054 は終値が建値から ATR x 0.5 有利なら翌朝に利確する。
どちらも新規と同時に、建値 ± 建てた足の ATR x tp_atr に指値を置き、足の中で触れたら利確する形(tp_order)を比べる。
reenter は利確の後、反対側のチャネルを割るまでは高値抜けを待たずに入り直す(E034)。
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
from fxsim.research.carry_harvest import NS_DAY, weekly
from fxsim.research.carry_loose import last_year
from fxsim.strategies import STRATEGIES

B049 = {"n_in": 10, "n_out": 5, "min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
B054 = {"n_in": 10, "n_out": 5, "min_carry": 0.01, "c2r_min": 0.2, "units": 2000, "margin_cap": 0.5}
GRID = [("B049 そのまま(利確なし)", B049)]
for k in (1.0, 1.5, 2.0, 3.0, 5.0):
    for re in (False, True):
        GRID.append((f"B049 + 指値 ATR x {k:g}" + ("・入り直す" if re else ""),
                     {**B049, "tp_atr": k, "tp_order": True, "reenter": re}))
GRID.append(("B054 そのまま(終値で ATR x 0.5・入り直す)", {**B054, "tp_atr": 0.5, "reenter": True}))
for k in (0.5, 1.0, 1.5, 2.0):
    GRID.append((f"B054 + 指値 ATR x {k:g}・入り直す", {**B054, "tp_atr": k, "tp_order": True, "reenter": True}))


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, p) in enumerate(GRID):
        s = STRATEGIES.create("main_carry_breakout", p)
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        row = {"stage": stage, "market": name, "k": k, **metrics(bt), **weekly(bt),
               "per_year": len(bt.trades) / years, "tp": s.tp_count}
        if name == "real_recent":
            row.update(last_year(bt))
        out.append(row)
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD", "ランダム 8 本 シャープ・最大 DD(中央・最悪)",
            "古い実データ シャープ / 年率 / 最大 DD", "古いランダム 16 本 シャープ(中央)・最大 DD(最悪)",
            "取引 / 年・利確の数(直近)", "利確のある週(直近 / 古い)",
            "直近 1 年: 取引 / 週・利確 / 週・損益(円)", "ロスカット・破綻"]
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
              f"{a['per_year']:.0f}・{a['tp']} | {a['win_weeks']:.0%} / {b['win_weeks']:.0%} | "
              f"{a['ly_trades_w']:.2f}・{a['ly_wins_w']:.2f}・{a['ly_pnl']:+,.0f} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
