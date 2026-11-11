"""E019: 本命 2 本目(B046、main_carry_trend)の部品ごとの効き目と、10 ペアを 1 つの口座で持ったときの成績。

    python -m fxsim.research.carry_trend_parts

相場とブローカーは E017 と同じ(real_pair の日足 10 ペア x 5 期間、OANDA)。
部品を 1 つずつ外した版と、方向でたらめ版を回す。
「10 ペアの口座」は、ペアごとの損益(各ペアは同じ 100 万円の口座で、損切りまでで 0.5% の量)を
日ごとに足したもの。
量は各ペアの口座の有効証拠金から決まるので、1 つの口座で 10 ペアを同時に持つ場合の近似(証拠金の取り合いは見ない)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import compute_metrics, daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS, RANGES
from fxsim.strategies import STRATEGIES

# 版: (名前, 戦略の上書き, ガードの上書き, 日足の約定の時, 方向でたらめ版か)
VARIANTS = [
    ("B046(全部)", {}, {}, 2, False),
    ("方向でたらめ版", {}, {}, 2, True),
    ("損切りをサーバーの逆指値(足の中で刺さる)", {}, {"close_stop": False}, 2, False),
    ("損切りを bid / ask の終値で判定", {}, {"stop_price": "side"}, 2, False),
    ("スプレッドのゲートなし", {}, {"max_spread_mult": 0.0}, 2, False),
    ("200 日平均の保険なし(純粋なキャリー)", {"trend_n": 0}, {}, 2, False),
    ("約定がロールオーバー(0 時)、ゲートあり", {}, {}, 0, False),
    ("約定がロールオーバー(0 時)、ゲートなし", {}, {"max_spread_mult": 0.0}, 0, False),
    ("急変用の逆指値を半分の距離(ATR x 4.5)", {}, {"emergency_stop_mult": 1.5}, 2, False),
    ("量を 2 倍(損切りまでで 1%)", {}, {"risk_per_trade": 0.01}, 2, False),
    ("量を 4 倍(損切りまでで 2%)", {}, {"risk_per_trade": 0.02}, 2, False),
]


def job(arg):
    pair, (start, end) = arg
    b = load_cases().behaviors["B046"]
    base = b.strategy_params()
    broker = BrokerConfig.preset("oanda", (b.broker_spec.get("overrides") or {}))
    mds = {}
    out = []
    for name, sp, gp, hour, null in VARIANTS:
        if hour not in mds:
            mds[hour] = MARKETS.create("real_pair", {"source": {"path": f"db:{pair}/D1", "start": start, "end": end,
                                                                "daily_fill_hour": hour}}).generate(seed=1)
        params = {**base, **sp, "guard": {**base.get("guard", {}), **gp}}
        if null:
            s = STRATEGIES.create("main_direction_null", {"inner": b.strategy, "inner_params": params,
                                                          "timeframe": 1440})
        else:
            s = STRATEGIES.create(b.strategy, params)
        bt = build_backtester(mds[hour], s, broker, EngineConfig(**b.engine), seed=1)
        bt.run()
        m = compute_metrics(bt)
        day_t, eq = daily_equity(bt)
        inner = s.inner if null else s
        bars = mds[hour].bars(1440)
        em = wick = 0
        for t in bt.trades:              # 急変用の逆指値(サーバーの sl)の約定と、その足の終値が内側に戻ったか(ひげだけ)
            if t.reason == "sl":
                k = int(np.searchsorted(bars.t, t.exit_t, "right")) - 1
                em += 1
                wick += int((bars.close[k] - t.exit_price) * t.side > 0)
        out.append(dict(name=name, pair=pair, start=start, pnl=m["final_equity"] - 1e6, swap=m["costs"]["swap"],
                        trades=m["n_trades"], max_dd=m["max_dd"], blocked=dict(inner.blocked),
                        emergency=em, wick_only=wick, days=(day_t // 86_400_000_000_000).tolist(),
                        eq=(eq - 1e6).tolist()))
    return out


def _t(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std() > 0 else 0.0


def account(rows: list[dict]) -> dict:
    """期間ごとに 10 ペアの損益を日ごとに足した口座(初期 100 万円)の年率・最大ドローダウン。"""
    out = {}
    for start in sorted({r["start"] for r in rows}):
        rs = [r for r in rows if r["start"] == start]
        days = sorted({d for r in rs for d in r["days"]})
        idx = {d: k for k, d in enumerate(days)}
        tot = np.zeros(len(days))
        for r in rs:
            v = np.full(len(days), np.nan)
            v[[idx[d] for d in r["days"]]] = r["eq"]
            # 前の値で埋める(始まる前は 0)
            last = 0.0
            for k in range(len(v)):
                last = v[k] if np.isfinite(v[k]) else last
                v[k] = last
            tot += v
        eq = 1e6 + tot
        years = (days[-1] - days[0]) / 365.25
        peak = np.maximum.accumulate(eq)
        out[start] = {"cagr": (eq[-1] / 1e6) ** (1 / years) - 1, "max_dd": float(((peak - eq) / peak).max())}
    return out


def report(rows: list[dict]) -> str:
    key = lambda r: (r["pair"], r["start"])  # noqa: E731
    full = sorted([r for r in rows if r["name"] == VARIANTS[0][0]], key=key)
    lines = ["| 版 | 平均(円 / 6 年) | 中央値 | プラス | t | 全部との差の t | スワップの平均 | 最悪のパターン "
             "| 取引 / 6 年 | 急変用の逆指値の約定(うちひげだけ) | 10 ペアの口座の年率(5 期間の平均) "
             "| 口座の最大 DD(最悪の期間) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, *_ in VARIANTS:
        rs = sorted([r for r in rows if r["name"] == name], key=key)
        p = np.array([r["pnl"] for r in rs])
        diff = [a["pnl"] - b["pnl"] for a, b in zip(rs, full, strict=True)]
        acc = account(rs)
        lines.append(f"| {name} | {p.mean():+,.0f} | {np.median(p):+,.0f} | {(p > 0).mean():.0%} | {_t(p):+.2f} | "
                     f"{_t(diff):+.2f} | {np.mean([r['swap'] for r in rs]):+,.0f} | {p.min():+,.0f} | "
                     f"{np.mean([r['trades'] for r in rs]):.1f} | "
                     f"{sum(r['emergency'] for r in rs)}({sum(r['wick_only'] for r in rs)}) | "
                     f"{np.mean([a['cagr'] for a in acc.values()]):+.2%} | "
                     f"{max(a['max_dd'] for a in acc.values()):.1%} |")
    acc = account(full)
    lines.append("")
    lines.append("B046 の 10 ペアの口座、期間ごと: " + ", ".join(
        f"{k[:4]} 年率 {v['cagr']:+.2%} / 最大 DD {v['max_dd']:.1%}" for k, v in acc.items()))
    blocked = {}
    for r in full:
        for k, v in r["blocked"].items():
            blocked[k] = blocked.get(k, 0) + v
    lines.append("B046 の止めた回数(50 パターンの合計): " + ", ".join(f"{k} {v}" for k, v in blocked.items() if v))
    return "\n".join(lines)


def main() -> None:
    args = [(p, w) for p in PAIRS for w in RANGES["D1"].windows]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(job, args):
            rows += r
    print(report(rows))


if __name__ == "__main__":
    main()
