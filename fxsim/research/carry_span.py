"""E030: スワップ狙い + ブレイクアウト(B047 の形)の時間の幅(入りと出のチャネルの長さ)を 1 週間〜3 年で変える。

    python -m fxsim.research.carry_span

入り = 直近 span 日の高値(売りは安値)を終値が抜ける、出 = 反対側の span x 1/2(または 1/4)日を割る。
日足で数える(1 週 = 5 本、1 か月 = 21 本、1 年 = 250 本)。週足・月足のチャネルは日足の長いチャネルと
同じ判定なので、日足だけで回す。
ほかは B047 のまま(金利差 2%・c2r 0.4、2,000 通貨、証拠金の上限 50%、急変用の逆指値 ATR x 9、口座の床)。
相場・口座は E026 と同じ。3 年のチャネルは最初の 3 年は入れないので、どの設定も最初の 3 年 1 か月を除いた期間で比べる。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.research.carry_breakout_opt import markets
from fxsim.strategies import STRATEGIES

NS_DAY = 86_400_000_000_000
SKIP_DAYS = 365 * 3 + 31
SPANS = [("1 週", 5), ("2 週", 10), ("3 週", 15), ("4 週", 20), ("5 週", 25), ("2 か月", 42), ("3 か月", 63),
         ("4 か月", 84), ("5 か月", 105), ("6 か月", 126), ("1 年", 250), ("2 年", 500), ("3 年", 750)]
EXIT = [("1/2", 0.5), ("1/4", 0.25)]
B047 = {"min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
GRID = [(f"{sn}・出 {en}", {**B047, "n_in": n, "n_out": max(2, round(n * f))}) for sn, n in SPANS for en, f in EXIT]


def measure(bt) -> dict:
    t, eq = daily_equity(bt)
    t0 = int(t[0]) + SKIP_DAYS * NS_DAY
    k = int(np.searchsorted(t, t0))
    t, eq = t[k:], eq[k:]
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    years = (t[-1] - t[0]) / (365.25 * NS_DAY)
    peak = np.maximum.accumulate(eq)
    trades = [x for x in bt.trades if x.entry_t >= t0]
    days = [(x.exit_t - x.entry_t) / NS_DAY for x in trades]
    pnl = sum(x.pnl for x in trades)
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(260)) if len(r) > 2 and r.std() > 0 else 0.0,
            "cagr": float((max(eq[-1], 1.0) / eq[0]) ** (1 / years) - 1), "dd": float(((peak - eq) / peak).max()),
            "per_year": len(trades) / years, "hold": float(np.median(days)) if days else 0.0,
            "swap_share": float(sum(x.swap for x in trades) / pnl) if pnl > 0 else float("nan"),
            "bad": bool(bt.stats.get("losscut", 0) or bt.ruined)}


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
        out.append({"stage": stage, "market": name, "k": k, **measure(bt)})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["入りの幅・出", "点数(直近の実データとランダムの半々)", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率 / 最大 DD(中央・最悪)",
            "取引 / 年・持つ日数の中央・スワップの割合(直近 / 古い実データ)", "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def rnd(x):
        med = {f: float(np.median([r[f] for r in x])) for f in ("sharpe", "cagr", "dd")}
        return f"{med['sharpe']:+.2f} / {med['cagr']:+.1%} / {med['dd']:.0%}・{max(r['dd'] for r in x):.0%}"

    def share(r):
        return "-" if not np.isfinite(r["swap_share"]) else f"{r['swap_share']:.0%}"
    for k, (label, _) in enumerate(GRID):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = [r for r in rs if r["market"] == "real_recent"][0]
        b = [r for r in rs if r["market"] == "real_old"][0]
        score = 0.5 * a["sharpe"] + 0.5 * float(np.median([r["sharpe"] for r in ri]))
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{a['per_year']:.1f}・{a['hold']:.0f} 日・{share(a)} / "
              f"{b['per_year']:.1f}・{b['hold']:.0f} 日・{share(b)} | "
              f"{sum(r['bad'] for r in rs)}/{len(rs)} |")


if __name__ == "__main__":
    main()
