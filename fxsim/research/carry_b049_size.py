"""E056: B049 の量と証拠金の上限を変えたときの年率・最大 DD・ロスカット(E026 と同じ 26 本の相場)。

    python -m fxsim.research.carry_b049_size

利用者の問い(2026-11-13): 10 万円・レバレッジ 25 倍の口座で、証拠金を「予想の DD の 1 割手前」(直近なら 70% くらい)まで
使うとどうなるか。証拠金の上限だけを上げても量が 2,000 通貨のままなら、同時に持つペアが増えるだけ。
だから 1 ペアの量も振る。
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

B049 = {"n_in": 10, "n_out": 5, "min_carry": 0.02, "c2r_min": 0.4}
SIZES = [(units, cap) for units in (2000, 4000, 6000, 10000) for cap in (0.5, 0.7)]


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for units, cap in SIZES:
        s = STRATEGIES.create("main_carry_breakout", {**B049, "units": units, "margin_cap": cap})
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        out.append({"stage": stage, "market": name, "units": units, "cap": cap, **metrics(bt)})
    return out


def _pick(rows, stage: str, real: bool) -> list[dict]:
    return [r for r in rows if r["stage"] == stage and r["market"].startswith("real") == real]


def main() -> None:
    with ProcessPoolExecutor(10) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    print("| 1 ペアの量 / 証拠金の上限 | 直近の実データ 2014〜2026 年率 / 最大 DD / 10 万円が 12 年後 "
          "| 並べ替え 8 本 年率 / 最大 DD(中央・最悪) | 古い実データ 1996〜2014 年率 / 最大 DD "
          "| 古い並べ替え 16 本 年率 / 最大 DD(中央・最悪)/ プラス | ロスカット・破綻(26 本) |")
    print("|---|---:|---:|---:|---:|---:|")
    for units, cap in SIZES:
        rs = [r for r in rows if r["units"] == units and r["cap"] == cap]
        rr, ri = _pick(rs, "is", True)[0], _pick(rs, "is", False)
        ro, rq = _pick(rs, "oos", True)[0], _pick(rs, "oos", False)
        bad = sum(r["losscut"] > 0 or r["ruined"] for r in rs)
        print(f"| {units:,} 通貨 / {cap:.0%} | {rr['cagr']:+.1%} / {rr['dd']:.0%} / {rr['final'] / 1e4:.1f} 万円 | "
              f"{np.median([r['cagr'] for r in ri]):+.1%} / {np.median([r['dd'] for r in ri]):.0%}・"
              f"{max(r['dd'] for r in ri):.0%} | {ro['cagr']:+.1%} / {ro['dd']:.0%} | "
              f"{np.median([r['cagr'] for r in rq]):+.1%} / {np.median([r['dd'] for r in rq]):.0%}・"
              f"{max(r['dd'] for r in rq):.0%} / {np.mean([r['cagr'] > 0 for r in rq]):.0%} | {bad}/{len(rs)} |")


if __name__ == "__main__":
    main()
