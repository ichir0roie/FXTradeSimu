"""E023: 基本の戦略それぞれが、どの局面で効くか(スワップを受け取る向きだけ、日足、主要 10 ペア x 5 期間、OANDA)。

    python -m fxsim.research.regime_map

戦略は日足の動作ケース(B004 など、動作ケースの設定のまま)と、日中向けの基本の戦略を日足で回したもの(既定のパラメータ)。
どれも EngineConfig.swap_side_only(全部の動作ケースの決まり)と、同じ決まりの方向でたらめ版を対にして回す。
取引ごとに、入る判断をした足(確定した日足)の局面を付ける(どれも因果的。区分は E013 で結果を見る前に決めたもの):
    流れ     60 日の対数リターン / (20 日ボラ x √60) を、建てた向きで見る: 順行(> +0.5)/ 中立 / 逆行(< −0.5)
    一方向   20 日の効率比(|20 日の動き| / 日々の動きの絶対値の和): もみ合い(< 0.2)/ 中 / 一方向(> 0.4)
    ボラ     20 日ボラ / 250 日ボラ: 静か(< 0.8)/ 中 / 荒い(> 1.2)
    金利差   受け取れる年率(業者の取り分を引いた後): 小(< 1%)/ 中(1〜3%)/ 大(> 3%)
数字は 1 取引あたりの円(1 万通貨など動作ケースの量、スワップ込み)。同じ区分の方向でたらめ版の 1 取引あたりの円と並べる。
区分ごとの t は取引単位(同じ経路の取引は独立でないので、目安。E001)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import compute_metrics
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa: F403
from fxsim.data import load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS, RANGES
from fxsim.strategies import STRATEGIES

RECV = 0.0044
BEHAVIORS = ["B004", "B007", "B008", "B010", "B016", "B018", "B033", "B034", "B035", "B045", "B046"]
BASICS = ["bollinger_reversion", "rsi_reversion", "stochastic_reversion", "donchian_breakout", "macd_cross",
          "adx_trend",
          "ichimoku", "perfect_order", "heikin_ashi", "kalman_trend", "online_logit", "regime_adaptive"]
FEATURES = {"流れ": ("逆行", "中立", "順行"), "一方向": ("もみ合い", "中", "一方向"), "ボラ": ("静か", "中", "荒い"),
            "金利差": ("小", "中", "大")}


def specs():
    book = load_cases()
    out = []
    for bid in BEHAVIORS:
        b = book.behaviors[bid]
        out.append((f"{bid} {b.strategy}", b.strategy, b.strategy_params(), b.engine, b.null_strategy()))
    for name in BASICS:
        out.append((f"{name}(日足)", name, {"timeframe": 1440}, {"units": 10000}, "direction_null"))
    return out


def features(bars, pair: str):
    c = bars.close
    lc = np.log(c)
    r = np.diff(lc, prepend=lc[0])
    n = len(c)
    vol20 = np.full(n, np.nan)
    vol250 = np.full(n, np.nan)
    for i in range(20, n):
        vol20[i] = r[i - 19:i + 1].std()
    for i in range(250, n):
        vol250[i] = r[i - 249:i + 1].std()
    trend = np.full(n, np.nan)
    trend[60:] = (lc[60:] - lc[:-60]) / (vol20[60:] * np.sqrt(60))
    er = np.full(n, np.nan)
    absr = np.abs(r)
    cs = np.cumsum(absr)
    er[20:] = np.abs(lc[20:] - lc[:-20]) / np.maximum(cs[20:] - cs[:-20], 1e-12)
    diff = load_rates(MULTI_RATES).pair_differential(pair, bars.t)
    return trend, er, vol20 / vol250, diff


def tag(side: int, k: int, f) -> dict:
    trend, er, vr, diff = f
    out = {}
    tz = trend[k] * side
    out["流れ"] = None if not np.isfinite(tz) else ("順行" if tz > 0.5 else "逆行" if tz < -0.5 else "中立")
    if np.isfinite(er[k]):
        out["一方向"] = "一方向" if er[k] > 0.4 else "もみ合い" if er[k] < 0.2 else "中"
    else:
        out["一方向"] = None
    out["ボラ"] = None if not np.isfinite(vr[k]) else ("荒い" if vr[k] > 1.2 else "静か" if vr[k] < 0.8 else "中")
    carry = side * diff[k] - RECV
    out["金利差"] = "大" if carry > 0.03 else "小" if carry < 0.01 else "中"
    return out


def job(arg):
    pair, (start, end) = arg
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{pair}/D1", "start": start, "end": end,
                                                 "daily_fill_hour": 2}}).generate(seed=1)
    bars = md.bars(1440)
    f = features(bars, pair.lower())
    rows = []
    for label, strat, params, engine, null_name in specs():
        broker = BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}})
        eng = {**engine, "swap_side_only": True}
        for kind in ("model", "null"):
            if kind == "model":
                s = STRATEGIES.create(strat, params)
            else:
                inner = {k: v for k, v in params.items() if k != "timeframe"}
                s = STRATEGIES.create(null_name, {"inner": strat, "inner_params": inner, "timeframe": 1440})
            bt = build_backtester(md, s, broker, EngineConfig(**eng), seed=1)
            bt.run()
            m = compute_metrics(bt)
            trades = []
            for t in bt.trades:
                k = int(np.searchsorted(bars.t, t.entry_t, "right")) - 2      # 入る判断をした確定足
                if k < 0:
                    continue
                trades.append({"pnl": float(t.pnl), **tag(t.side, k, f)})
            rows.append({"label": label, "kind": kind, "pair": pair, "start": start,
                         "pnl": float(m["final_equity"] - 1e6), "trades": trades})
    return rows


def _t(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else 0.0


def report(rows) -> str:
    labels = list(dict.fromkeys(r["label"] for r in rows))
    lines = ["## 全体(パターン = ペア x 6 年の期間、50 パターン)", "",
             "| 戦略 | 平均(万円 / 6 年) | プラス | t | 帰無(同じ決まり)との差の t | 取引 / 6 年 |",
             "|---|---:|---:|---:|---:|---:|"]
    key = lambda r: (r["pair"], r["start"])  # noqa: E731
    for lab in labels:
        m = sorted([r for r in rows if r["label"] == lab and r["kind"] == "model"], key=key)
        n = sorted([r for r in rows if r["label"] == lab and r["kind"] == "null"], key=key)
        p = np.array([r["pnl"] for r in m])
        d = [a["pnl"] - b["pnl"] for a, b in zip(m, n, strict=True)]
        lines.append(f"| {lab} | {p.mean() / 1e4:+.1f} | {(p > 0).mean():.0%} | {_t(p):+.2f} | {_t(d):+.2f} | "
                     f"{np.mean([len(r['trades']) for r in m]):.1f} |")
    for feat, cats in FEATURES.items():
        lines += ["", f"## 局面: {feat}(1 取引あたりの円。括弧は同じ区分の方向でたらめ版、取引数)", "",
                  "| 戦略 | " + " | ".join(cats) + " |", "|---|" + "---:|" * len(cats)]
        for lab in labels:
            cells = []
            for c in cats:
                mv = [t["pnl"] for r in rows if r["label"] == lab and r["kind"] == "model" for t in r["trades"]
                      if t[feat] == c]
                nv = [t["pnl"] for r in rows if r["label"] == lab and r["kind"] == "null" for t in r["trades"]
                      if t[feat] == c]
                if len(mv) < 10:
                    cells.append("-")
                    continue
                cells.append(f"{np.mean(mv):+,.0f}(帰無 {np.mean(nv) if nv else 0:+,.0f}、{len(mv)}、t {_t(mv):+.1f})")
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
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
