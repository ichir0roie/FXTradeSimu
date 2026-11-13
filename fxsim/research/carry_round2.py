"""E045: スワップ狙いの 2 本(B049 = 高値抜け、B059 = EMA)に、4 つの案を足して口座の成績で比べる。

    python -m fxsim.research.carry_round2

    A  B059 に利確の指値(建値 ± ATR x 2)。入り直す・入り直さない
    B  急変用の逆指値を ATR x 9 から x 5・x 3 に近づける(E042: ATR x 3 で刺さるのは取引の 7%)
    C  ボラで量を決める(ATR / 終値が 0.7% のペアで 2,000 通貨、振れの大きいペアは減らす。1,000 通貨刻み、上限 4,000)
    D  B049 と B059 を半分ずつ(5 万円・1,000 通貨の口座を 2 つ持ち、有効証拠金を足す)
相場・口座・点数は E026 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ + 古いランダム 16 本、
10 万円、OANDA、約定は 2 時、swap_side_only)。
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
from fxsim.research.carry_harvest import NS_DAY
from fxsim.strategies import STRATEGIES

BASE = {"min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
BRK = ("main_carry_breakout", {**BASE, "n_in": 10, "n_out": 5})
EMA = ("main_carry_ema", {**BASE, "ema_fast": 5, "ema_slow": 20, "signal_every": 5})
TP = {"tp_atr": 2.0, "tp_order": True}
VOL = {"vol_target": 0.007}


def v(base, **extra):
    return [(base[0], {**base[1], **extra}, 1.0)]


def half(*bases):
    return [(b[0], {**b[1], "units": b[1]["units"] / len(bases)}, 1.0 / len(bases)) for b in bases]


# 名前 -> [(戦略, 引数, 口座の割合), ...]。2 つ以上なら別々の口座で回して有効証拠金を足す
GRID = [
    ("B049(高値抜け)", v(BRK)),
    ("B057(B049 + 利確 ATR x 2・入り直す)", v(BRK, **TP, reenter=True)),
    ("B059(EMA5/20・週に 1 回判定)", v(EMA)),
    ("A B059 + 利確 ATR x 2", v(EMA, **TP)),
    ("A B059 + 利確 ATR x 2・入り直す", v(EMA, **TP, reenter=True)),
    ("B B049・逆指値 ATR x 5", v(BRK, emergency_atr=5.0)),
    ("B B049・逆指値 ATR x 3", v(BRK, emergency_atr=3.0)),
    ("B B059・逆指値 ATR x 5", v(EMA, emergency_atr=5.0)),
    ("B B059・逆指値 ATR x 3", v(EMA, emergency_atr=3.0)),
    ("C B049・ボラで量", v(BRK, **VOL)),
    ("C B059・ボラで量", v(EMA, **VOL)),
    ("D B049 と B059 を半分ずつ", half(BRK, EMA)),
    ("D B057 と B059 + 利確・入り直す を半分ずつ", half((BRK[0], {**BRK[1], **TP, "reenter": True}),
                                            (EMA[0], {**EMA[1], **TP, "reenter": True}))),
]


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    out = []
    for k, (_, parts) in enumerate(GRID):
        eq, trades, bad, years, t = None, 0, False, 1.0, None
        for strat, p, share in parts:
            cfg = EngineConfig(initial_balance=100_000.0 * share, min_units=1000.0, swap_side_only=True)
            bt = build_backtester(md, STRATEGIES.create(strat, p), BrokerConfig.preset("oanda"), cfg, seed=1)
            bt.run()
            t, e = daily_equity(bt)
            eq = e if eq is None else eq + e
            trades += len(bt.trades)
            bad |= bool(bt.stats.get("losscut", 0) or bt.ruined)
            years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        r = np.diff(eq) / eq[:-1]
        r = r[np.isfinite(r)]
        peak = np.maximum.accumulate(eq)
        out.append({"stage": stage, "market": name, "k": k, "bad": bad, "per_year": trades / years,
                    "sharpe": float(r.mean() / r.std() * np.sqrt(260)) if len(r) > 2 and r.std() > 0 else 0.0,
                    "cagr": float((max(eq[-1], 1.0) / eq[0]) ** (1 / years) - 1),
                    "dd": float(((peak - eq) / peak).max())})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率・最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率・最大 DD(中央・最悪)", "年率 ÷ 最大 DD(ランダム 24 本の中央)",
            "取引 / 年(直近 / 古い)", "B049 以上のシャープ・年率 ÷ DD(ランダム 24 本)", "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def med(x, f):
        return float(np.median([r[f] for r in x]))

    def rnd(x):
        return (f"{med(x, 'sharpe'):+.2f} / {med(x, 'cagr'):+.1%}・{med(x, 'dd'):.0%}・"
                f"{max(r['dd'] for r in x):.0%}")

    def calmar(r):
        return r["cagr"] / r["dd"] if r["dd"] > 0 else 0.0
    ref = {r["market"]: r for r in rows if r["k"] == 0}
    for k, (label, _) in enumerate(GRID):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = next(r for r in rs if r["market"] == "real_recent")
        b = next(r for r in rs if r["market"] == "real_old")
        score = 0.5 * a["sharpe"] + 0.5 * med(ri, "sharpe")
        rr = ri + rq
        b1 = sum(r["sharpe"] >= ref[r["market"]]["sharpe"] for r in rr)
        b2 = sum(calmar(r) >= calmar(ref[r["market"]]) for r in rr)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{float(np.median([calmar(r) for r in rr])):.2f} | {a['per_year']:.1f} / {b['per_year']:.1f} | "
              f"{b1}/{len(rr)}・{b2}/{len(rr)} | {sum(r['bad'] for r in rs)}/{len(rs)} |")


if __name__ == "__main__":
    main()
