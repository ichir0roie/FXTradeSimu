"""E034: 本命の 2 週版(B049)に利確の刈り取りを足して、週に 1 回は利確する形に近づける。

    python -m fxsim.research.carry_harvest

戦略の下限(金利差 2%・c2r 0.4)とブレイクアウトの入りは B049 のまま。変えるのは
    tp_atr   終値が建値から ATR(建てた足)x k だけ有利なら利確(0 = 使わない)
    reenter  利確の後、反対側のチャネルを割るまでは高値抜けを待たずに入り直す
    幅       2 週(B049: 10 日抜けで入り 5 日で出る)と 1 週(5 日抜け・2 日で出る)
相場・口座・点数は E026 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ 1996〜2014-10 +
古いランダム 16 本、10 万円、OANDA、約定は 2 時、swap_side_only)。

週の数え方: 利確 = 損益(スワップ込み)がプラスの手仕舞い。週は手仕舞いの日の暦の週(月曜始まり、UTC)。
「建玉のある週」は、どれかのペアを 1 日でも持っていた週。
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

NS_DAY = 86_400_000_000_000
NS_WEEK = 7 * NS_DAY
MONDAY = np.datetime64("1970-01-05", "ns").astype(np.int64)      # 週の区切り(月曜 0 時 UTC)
BASE = {"min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
SPANS = [("2 週", {"n_in": 10, "n_out": 5}), ("1 週", {"n_in": 5, "n_out": 2})]
TPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
GRID = []
for sn, sp in SPANS:
    for k in TPS:
        for re in ((False,) if k == 0 else (False, True)):
            label = f"{sn}・" + ("利確なし" if k == 0 else f"ATR x {k:g} で利確" + ("・入り直す" if re else ""))
            GRID.append((label, {**BASE, **sp, "tp_atr": k, "reenter": re}))


def week(t) -> np.ndarray:
    return (np.asarray(t, dtype=np.int64) - MONDAY) // NS_WEEK


def weekly(bt) -> dict:
    t = bt.bars.t
    weeks = np.unique(week(t))
    wins = [x for x in bt.trades if x.pnl > 0]
    win_weeks = set(week([x.exit_t for x in wins]).tolist())
    held = set()
    for x in bt.trades:
        held.update(range(int(week([x.entry_t])[0]), int(week([x.exit_t])[0]) + 1))
    return {"wins_per_week": len(wins) / len(weeks), "win_weeks": len(win_weeks) / len(weeks),
            "held_weeks": len(held) / len(weeks),
            "win_of_held": len(win_weeks) / len(held) if held else 0.0,
            "win_rate": len(wins) / len(bt.trades) if bt.trades else 0.0,
            "hold": float(np.median([(x.exit_t - x.entry_t) / NS_DAY for x in bt.trades])) if bt.trades else 0.0}


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
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), **weekly(bt),
                    "per_year": len(bt.trades) / years})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪)",
            "古い実データ シャープ / 年率 / 最大 DD", "古いランダム 16 本 シャープ(中央)・最大 DD(最悪)",
            "取引 / 年・勝率・持つ日数(直近)",
            "利確 / 週(直近 / 古い / ランダム中央)",
            "利確のある週(全部の週に対して、直近 / 古い / ランダム中央)", "建玉のある週のうち利確のある週(直近 / 古い)",
            "ロスカット・破綻"]
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
              f"{med(ri, 'sharpe'):+.2f} / {med(ri, 'cagr'):+.1%} / {med(ri, 'dd'):.0%}・"
              f"{max(r['dd'] for r in ri):.0%} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | "
              f"{med(rq, 'sharpe'):+.2f}・{max(r['dd'] for r in rq):.0%} | "
              f"{a['per_year']:.0f}・{a['win_rate']:.0%}・{a['hold']:.0f} 日 | "
              f"{a['wins_per_week']:.2f} / {b['wins_per_week']:.2f} / {med(ri, 'wins_per_week'):.2f} | "
              f"{a['win_weeks']:.0%} / {b['win_weeks']:.0%} / {med(ri, 'win_weeks'):.0%} | "
              f"{a['win_of_held']:.0%} / {b['win_of_held']:.0%} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
