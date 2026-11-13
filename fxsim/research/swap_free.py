"""E046: スワップの向きを考えずに(両方向で)同じ信号を回すとどうなるか。払う向きだけ・受け取る向きだけと並べる。

    python -m fxsim.research.swap_free

信号は 5 つ(10 日抜け・55 日抜け・EMA5/20 の週次・EMA20/60・63 日モメンタムの週次)、向きは 4 つ
(受け取る向き + 下限 = B049 の形 / 受け取る向き・下限なし / 両方向 / 払う向きだけ)。
1 ペア 2,000 通貨・証拠金の上限 50%・急変用の逆指値 ATR x 9・口座の床は同じ。受け取る向きの 2 つは swap_side_only、
両方向・払う向きは外す。相場・点数は E026 と同じ(OANDA、約定は 2 時)。口座だけ 100 万円にする: 10 万円では両方向・
払う向きの版が数年で証拠金不足になって取引が止まり、信号の損益を比べられない
(年率と最大 DD は 10 万円の口座の 1/10 ほどに出る)。
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

BALANCE = 1_000_000.0
SIGNALS = [("10 日抜け・5 日で出る", {"signal": "break", "n_in": 10, "n_out": 5}),
           ("55 日抜け・20 日で出る", {"signal": "break", "n_in": 55, "n_out": 20}),
           ("EMA5/20・週次", {"signal": "ema", "ema_fast": 5, "ema_slow": 20, "signal_every": 5}),
           ("EMA20/60", {"signal": "ema", "ema_fast": 20, "ema_slow": 60}),
           ("63 日モメンタム・週次", {"signal": "mom", "mom_n": 63, "signal_every": 5})]
SIDES = [("受け取る向き + 下限", {"sides": "recv", "min_carry": 0.02, "c2r_min": 0.4}, True),
         ("受け取る向き・下限なし", {"sides": "recv", "min_carry": 0.0, "c2r_min": 0.0}, True),
         ("両方向", {"sides": "both", "c2r_min": 0.0}, False),
         ("払う向きだけ", {"sides": "pay", "c2r_min": 0.0}, False)]
GRID = [(f"{sn} / {dn}", {"units": 2000, "margin_cap": 0.5, **sp, **dp}, gate)
        for sn, sp in SIGNALS for dn, dp, gate in SIDES]


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    out = []
    for k, (_, p, gate) in enumerate(GRID):
        cfg = EngineConfig(initial_balance=BALANCE, min_units=1000.0, swap_side_only=gate)
        bt = build_backtester(md, STRATEGIES.create("main_trend_free", p), BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        swap = sum(x.swap for x in bt.trades)
        pnl = sum(x.pnl for x in bt.trades)
        win = float(np.mean([x.pnl > 0 for x in bt.trades])) if bt.trades else 0.0
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "per_year": len(bt.trades) / years,
                    "swap": swap / years, "move": (pnl - swap) / years, "win": win})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["信号 / 向き", "点数", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率・最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率・最大 DD(中央・最悪)",
            "値動き(スワップを除く)円 / 年: 直近・古い実データ・ランダム 24 本の平均と t",
            "スワップ 円 / 年(同じ並び)", "取引 / 年・勝率(直近)", "ロスカット・破綻"]
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
    for k, (label, _, _) in enumerate(GRID):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = next(r for r in rs if r["market"] == "real_recent")
        b = next(r for r in rs if r["market"] == "real_old")
        score = 0.5 * a["sharpe"] + 0.5 * med(ri, "sharpe")
        mv = [r["move"] for r in ri + rq]
        sw = [r["swap"] for r in ri + rq]
        bad = sum(bool(r["losscut"] or r["ruined"]) for r in rs)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{a['move']:+,.0f}・{b['move']:+,.0f}・{np.mean(mv):+,.0f}(t {tstat(mv):+.1f}) | "
              f"{a['swap']:+,.0f}・{b['swap']:+,.0f}・{np.mean(sw):+,.0f} | "
              f"{a['per_year']:.0f}・{a['win']:.0%} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
