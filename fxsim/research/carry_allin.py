"""E057: B049 を、資金に比例して(複利で)証拠金の限界近くまで張り続けたらどうなるか。生き残れる一番大きな張り方はどこか。

    python -m fxsim.research.carry_allin

利用者の問い(2026-11-13): 限界ぎりぎりまで、使える資金を使い切って投機し続けた場合。その中で生き残る最大の形。
B049 の決まりのまま、新規 1 本の必要証拠金を「その時の有効証拠金 x 割合」にする(margin_frac。量が資金と一緒に増減する)。
証拠金の上限(口座全体)も振る。OANDA の口座: レバレッジ 25 倍、有効証拠金が必要証拠金を割ったら全部ロスカット、ゼロカットなし。
相場は E026 と同じ 26 本。ロスカットの後も、残った資金で同じ決まりのまま続ける。
"""
# ruff: noqa: E501
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
# (新規 1 本の証拠金 / 有効証拠金, 口座全体の証拠金の上限)
SIZES = ([(f, 0.98) for f in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)]
         + [(0.33, 0.98), (0.49, 0.98), (0.98, 0.98)]
         + [(f, 0.5) for f in (0.05, 0.1, 0.2)])


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for f, cap in SIZES:
        s = STRATEGIES.create("main_carry_breakout", {**B049, "margin_frac": f, "margin_cap": cap})
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        out.append({"stage": stage, "market": name, "f": f, "cap": cap, **metrics(bt)})
    return out


def main() -> None:
    with ProcessPoolExecutor(10) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    print("| 新規 1 本の証拠金(資金の何割)/ 全体の上限 | 1 本の建玉(資金の何倍) | 直近の実データ 2014〜2026 年率 / 最大 DD / 10 万円が 12 年後 "
          "| 古い実データ 1996〜2014 年率 / 最大 DD / 10 万円が 19 年後 | 並べ替え 24 本 年率の中央 / 最大 DD の中央・最悪 "
          "| 並べ替え 24 本 12〜19 年後の資金(下から 10%・中央・上から 10%) | ロスカットのあった相場(26 本) | 元本割れ(26 本) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f, cap in SIZES:
        rs = [r for r in rows if r["f"] == f and r["cap"] == cap]
        rec = next(r for r in rs if r["market"] == "real_recent")
        old = next(r for r in rs if r["market"] == "real_old")
        rnd = [r for r in rs if not r["market"].startswith("real")]
        fin = np.array([r["final"] for r in rnd]) / 1e4
        lc = sum(r["losscut"] > 0 or r["ruined"] for r in rs)
        below = sum(r["final"] < 100_000 for r in rs)
        print(f"| {f:.0%} / {cap:.0%} | {f * 25:.1f} | {rec['cagr']:+.1%} / {rec['dd']:.0%} / {rec['final'] / 1e4:,.1f} 万円 "
              f"| {old['cagr']:+.1%} / {old['dd']:.0%} / {old['final'] / 1e4:,.1f} 万円 "
              f"| {np.median([r['cagr'] for r in rnd]):+.1%} / {np.median([r['dd'] for r in rnd]):.0%}・{max(r['dd'] for r in rnd):.0%} "
              f"| {np.quantile(fin, 0.1):,.1f}・{np.median(fin):,.1f}・{np.quantile(fin, 0.9):,.1f} 万円 | {lc}/26 | {below}/26 |")


if __name__ == "__main__":
    main()
