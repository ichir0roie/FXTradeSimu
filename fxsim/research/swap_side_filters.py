"""E036: スワップを受け取る向きだけを方針にした順張りの、負けどころと対策(USDJPY、参考に主要 10 ペア、OANDA)。

    python -m fxsim.research.swap_side_filters [--workers N]

順張り 8 本(E033 と同じ、既定のパラメータ)を、方針(EngineConfig.swap_side_only)と E035 の対策を掛けて回し、
同じ方針・同じ対策の方向でたらめ版と対にする。方針の下の方向でたらめ版は「信号の時刻に、信号の向きに関係なく受け取る向きに入る」。
    短期(1 分・5 分・15 分): 持ち越さない(ロールオーバー前にたたむ = 1')を土台に、両方向 / 受け取る向きだけ、と対策
    1 時間・4 時間: 受け取る向きだけ(持ち越す)を基準に、対策 1(ロールオーバーは新規なし)・4・6・5a と組み合わせ
    日足: 受け取る向きだけを基準に、4・5a(時間帯・曜日は日足 1 本の約定時刻では意味が無い)
    主要 10 ペア(参考、記録の壊れた AUDJPY を除く 9 ペア): 4 時間・日足で、
        両方向 / 受け取る向きだけ / 受け取る向きだけ + 4
分解: スワップ(pip に直す)、ドリフト分(期間の平均の値動き x 持っていた時間 x 向き、pip)、
値動きの優位 = 素 − ドリフト分。
負けどころは、USDJPY の 1 時間・4 時間・日足の方針の基準の取引に E033 の区分と、受け取る金利差の区分を付けて見る。
"""
from __future__ import annotations

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.data import load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.experiments.config import default_workers
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS
from fxsim.research.loss_filters import ER, NO_FRIDAY, OUT_ROLLOVER, VOL, schedule
from fxsim.research.loss_timing import FEATURES, JA, JST, NS_MIN, TREND, bar_features, tag
from fxsim.strategies import STRATEGIES

FLAT = {"hours": OUT_ROLLOVER, "outside": "close"}
V_SHORT = [
    ("両方向 + 1'", FLAT, {}, False),
    ("受取 + 1'", FLAT, {}, True),
    ("受取 + 1' + 4", FLAT, {"max_vol_ratio": VOL}, True),
    ("受取 + 1' + 6", {**FLAT, "days": NO_FRIDAY}, {}, True),
    ("受取 + 1' + 4 + 6", {**FLAT, "days": NO_FRIDAY}, {"max_vol_ratio": VOL}, True),
]
V_HOUR = [
    ("両方向", {}, {}, False),
    ("受取", {}, {}, True),
    ("受取 + 1", {"hours": OUT_ROLLOVER}, {}, True),
    ("受取 + 4", {}, {"max_vol_ratio": VOL}, True),
    ("受取 + 6", {"days": NO_FRIDAY}, {}, True),
    ("受取 + 5a", {}, {"max_er": ER}, True),
    ("受取 + 1 + 4 + 6", {"hours": OUT_ROLLOVER, "days": NO_FRIDAY}, {"max_vol_ratio": VOL}, True),
    ("受取 + 1' + 4 + 6", {**FLAT, "days": NO_FRIDAY}, {"max_vol_ratio": VOL}, True),
]
V_DAY = [
    ("両方向", {}, {}, False),
    ("受取", {}, {}, True),
    ("受取 + 4", {}, {"max_vol_ratio": VOL}, True),
    ("受取 + 5a", {}, {"max_er": ER}, True),
]
V_PAIRS = [V_DAY[0], V_DAY[1], V_DAY[2]]
H1_SPAN = ("2010-10-06", "2026-11-11")
D1_SPAN = ("1996-01-01", "2026-11-10")
# (ラベル, 戦略の足, 相場の DB の足, 開始, 終了, 版)
USDJPY_CASES = [
    ("1 分", 1, "M1", "2026-08-05", "2026-11-11", V_SHORT),
    ("5 分", 5, "M5", "2025-07-09", "2026-11-11", V_SHORT),
    ("15 分", 15, "M5", "2025-07-09", "2026-11-11", V_SHORT),
    ("1 時間", 60, "H1", *H1_SPAN, V_HOUR),
    ("4 時間", 240, "H1", *H1_SPAN, V_HOUR),
    ("日足", 1440, "D1", *D1_SPAN, V_DAY),
]
PAIR_CASES = [("4 時間", 240, "H1", *H1_SPAN, V_PAIRS), ("日足", 1440, "D1", *D1_SPAN, V_PAIRS)]
DETAIL = ("1 時間", "4 時間", "日足")       # 負けどころを見る足(方針の基準 = 受取)
# 相場の DB の AUDJPY は、記録のスプレッドが 140〜460 pip の足を含む(H1 で 1,940 点、D1 で 88 点)。約定した瞬間に
# 4 万円台の負けとロスカットが出て結果が壊れるので外す(E036 で見つけた。データの直しは別)
BROKEN = {"AUDJPY"}
REF_PAIRS = [p for p in PAIRS if p not in BROKEN]
CARRY = ("小", "中", "大")


def _market(pair, data_tf, start, end):
    src = {"path": f"db:{pair}/{data_tf}", "start": start, "end": end}
    if data_tf == "D1":
        src["daily_fill_hour"] = 2
    return MARKETS.create("real_pair", {"source": src}).generate(seed=1)


def job(arg):
    pair, label, tf, data_tf, start, end, variants, method = arg
    with contextlib.suppress(OSError):
        os.nice(10)
    md = _market(pair, data_tf, start, end)
    bars = md.bars(tf)
    f = bar_features(bars)
    lc = np.log(bars.close)
    mu = (lc[-1] - lc[0]) / float(bars.t[-1] - bars.t[0])
    broker = BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}})
    diff = bars.rate_diff if bars.rate_diff is not None else \
        load_rates(MULTI_RATES).pair_differential(pair.lower(), bars.t)
    ok = np.array([broker.swap_annual(1, d) > 0 or broker.swap_annual(-1, d) > 0 for d in diff])
    out = []
    for vlabel, sched, filt, swap in variants:
        for kind in ("model", "null"):
            inner, inner_params = (method, {}) if kind == "model" else ("direction_null", {"inner": method})
            s = STRATEGIES.create("entry_filter", {"inner": inner, "inner_params": inner_params, "timeframe": tf,
                                                   **filt})
            bt = build_backtester(md, s, broker, EngineConfig(units=10_000, swap_side_only=swap), seed=1,
                                  schedule=schedule(sched))
            bt.run()
            raw = [t for t in bt.trades if t.reason != "end"]
            pv = [(t.pnl - t.swap + t.commission) / t.pips for t in raw if abs(t.pips) > 0.5]
            pipval = float(np.median(pv)) if pv else 100.0          # 1 pip の円(1 万通貨)
            trades = []
            for t in raw:
                k = int(np.searchsorted(bars.t, t.entry_t, "right")) - 2
                if k < 0:
                    continue
                d = {"pnl": float(t.pnl), "pct": float(t.pnl / (abs(t.units) * t.entry_price) * 100),
                     "pips": float(t.pips), "gross": float(t.gross_pips), "cost": float(t.cost_pips),
                     "swap": float(t.swap / pipval), "pipval": pipval,
                     "drift": float(t.side * mu * (t.exit_t - t.entry_t) * t.entry_price / md.pip),
                     "week": int((t.entry_t + JST) // (7 * 1440 * NS_MIN))}
                if pair == "USDJPY" and label in DETAIL and vlabel == "受取":
                    d.update(tag(t, k, tf, f))
                    carry = broker.swap_annual(t.side, diff[k])
                    d["金利差"] = "大" if carry > 0.03 else "小" if carry < 0.01 else "中"
                trades.append(d)
            out.append({"pair": pair, "case": label, "method": method, "variant": vlabel, "kind": kind,
                        "trades": trades, "ok": float(ok.mean())})
    return out


def _t(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else 0.0


def _weeks(tr) -> dict:
    w: dict[int, float] = {}
    for t in tr:
        w[t["week"]] = w.get(t["week"], 0.0) + t["pnl"]
    return w


def _diff_t(tr, nu) -> float:
    a, b = _weeks(tr), _weeks(nu)
    keys = sorted(set(a) | set(b))
    return _t([a.get(k, 0.0) - b.get(k, 0.0) for k in keys])


def stats(tr) -> dict:
    if not tr:
        return {"n": 0}
    pnl = np.array([t["pnl"] for t in tr])
    pct = np.array([t["pct"] for t in tr])
    pips = np.array([t["pips"] for t in tr])
    win = pnl > 0
    wp, lp = pips[win], pips[~win]
    swap = np.mean([t["swap"] for t in tr])
    gross = np.mean([t["gross"] for t in tr])
    drift = np.mean([t["drift"] for t in tr])
    return {"n": len(tr), "win": win.mean(), "win_pct": pct[win].mean() if win.any() else 0.0,
            "loss_pct": pct[~win].mean() if (~win).any() else 0.0,
            "be": float(-lp.mean() / (wp.mean() - lp.mean())) if len(wp) and len(lp) and lp.mean() < 0 else np.nan,
            "exp": pips.mean() + swap, "gross": gross, "cost": np.mean([t["cost"] for t in tr]), "swap": swap,
            "drift": drift, "move": gross - drift, "ex_swap": pips.mean(), "total": pnl.sum() / 1e4,
            "t": _t(list(_weeks(tr).values())),
            "yen": pnl.mean(), "yen_ex": np.mean([t["pnl"] - t["swap"] * t["pipval"] for t in tr]),
            "yen_drift": np.mean([t["drift"] * t["pipval"] for t in tr]),
            "yen_move": np.mean([(t["gross"] - t["drift"]) * t["pipval"] for t in tr])}


def _pool(rows, pair, case, variant, kind, methods=TREND):
    return [t for r in rows if r["pair"] == pair and r["case"] == case and r["variant"] == variant
            and r["kind"] == kind and r["method"] in methods for t in r["trades"]]


def table_usdjpy(rows) -> list[str]:
    out = []
    for label, _tf, _d, _s, _e, variants in USDJPY_CASES:
        ok = next(r["ok"] for r in rows if r["pair"] == "USDJPY" and r["case"] == label)
        out += [f"### {label}(受け取れる向きがあった足の割合 {ok:.0%})", "",
                "| 方針・対策 | 取引 | 勝率 | 勝ちの平均 | 負けの平均 | 期待値(pip) | スワップ(pip) | "
                "スワップを除く(pip) | ドリフト分(pip) | 値動きの優位(pip) | 帰無の期待値 / スワップを除く(pip) | "
                "帰無との差の t | 合計(万円) | 週の t |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v, *_ in variants:
            tr, nu = _pool(rows, "USDJPY", label, v, "model"), _pool(rows, "USDJPY", label, v, "null")
            s, sn = stats(tr), stats(nu)
            if s["n"] == 0:
                out.append(f"| {v} | 0 |" + " - |" * 12)
                continue
            out.append(f"| {v} | {s['n']} | {s['win']:.0%} | {s['win_pct']:+.3f}% | {s['loss_pct']:+.3f}% | "
                       f"{s['exp']:+.2f} | {s['swap']:+.2f} | {s['ex_swap']:+.2f} | {s['drift']:+.2f} | "
                       f"{s['move']:+.2f} | {sn.get('exp', 0):+.2f} / {sn.get('ex_swap', 0):+.2f} | "
                       f"{_diff_t(tr, nu):+.1f} | {s['total']:+.1f} | {s['t']:+.1f} |")
        out.append("")
    return out


def table_methods(rows) -> list[str]:
    cols = [("1 時間", "受取"), ("1 時間", "受取 + 1"), ("4 時間", "受取"), ("4 時間", "受取 + 1"), ("日足", "受取"),
            ("日足", "受取 + 4")]
    out = ["| 手法 | " + " | ".join(f"{c} {v}" for c, v in cols) + " |", "|---|" + "---:|" * len(cols)]
    for m in TREND:
        cells = []
        for c, v in cols:
            tr, nu = _pool(rows, "USDJPY", c, v, "model", [m]), _pool(rows, "USDJPY", c, v, "null", [m])
            s = stats(tr)
            cells.append(f"{s['exp']:+.1f} / 除 {s['ex_swap']:+.1f}(差の t {_diff_t(tr, nu):+.1f}、{s['n']})"
                         if s["n"] else "-")
        out.append(f"| {JA[m]} | " + " | ".join(cells) + " |")
    return out


def table_where(rows) -> list[str]:
    feats = {**{k: FEATURES[k] for k in ("時間帯", "曜日", "ボラ", "一方向", "流れ", "持つ長さ")}, "金利差": CARRY}
    out = []
    for feat, cats in feats.items():
        out += [f"### {feat}", "", "| 足 | " + " | ".join(cats) + " |", "|---|" + "---:|" * len(cats)]
        for case in DETAIL:
            if feat == "時間帯" and case == "日足":
                continue
            tr = _pool(rows, "USDJPY", case, "受取", "model")
            loss_total = -sum(t["pnl"] for t in tr if t["pnl"] < 0)
            cells = []
            for c in cats:
                sub = [t for t in tr if t.get(feat) == c]
                if len(sub) < 20:
                    cells.append(f"-({len(sub)})")
                    continue
                s = stats(sub)
                share = -sum(t["pnl"] for t in sub if t["pnl"] < 0) / max(loss_total, 1e-9) / (len(sub) / len(tr))
                cells.append(f"{s['win']:.0%} / {s['exp']:+.1f}(除 {s['ex_swap']:+.1f})/ 負け {share:.2f}x")
            out.append(f"| {case} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def table_pairs(rows) -> list[str]:
    out = []
    for label, *_rest in PAIR_CASES:
        out += [f"### {label}", "",
                "| ペア | 受け取れる足の割合 | 版 | 取引 | 損益(円 / 取引) | スワップを除く | ドリフト分 | "
                "値動きの優位 | 帰無のスワップを除く | 帰無との差の t |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        summary = {v: [] for v, *_ in V_PAIRS}
        for pair in REF_PAIRS:
            ok = next(r["ok"] for r in rows if r["pair"] == pair and r["case"] == label)
            for v, *_ in V_PAIRS:
                tr, nu = _pool(rows, pair, label, v, "model"), _pool(rows, pair, label, v, "null")
                s, sn = stats(tr), stats(nu)
                if s["n"] == 0:
                    continue
                d = _diff_t(tr, nu)
                summary[v].append((s["yen_ex"], s["yen_move"], s["yen_ex"] - sn["yen_ex"], d))
                out.append(f"| {pair} | {ok:.0%} | {v} | {s['n']} | {s['yen']:+,.0f} | {s['yen_ex']:+,.0f} | "
                           f"{s['yen_drift']:+,.0f} | {s['yen_move']:+,.0f} | {sn['yen_ex']:+,.0f} | {d:+.1f} |")
        out += ["", "| 版 | スワップを除いてプラスのペア | 値動きの優位がプラスのペア | "
                "帰無(スワップを除く)より上のペア | 帰無との差の t ≥ 2 のペア |", "|---|---:|---:|---:|---:|"]
        for v, xs in summary.items():
            n = len(xs)
            out.append(f"| {v} | {sum(x[0] > 0 for x in xs)}/{n} | {sum(x[1] > 0 for x in xs)}/{n} | "
                       f"{sum(x[2] > 0 for x in xs)}/{n} | {sum(x[3] >= 2 for x in xs)}/{n} |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    args = [("USDJPY", *c, m) for c in USDJPY_CASES for m in TREND]
    args += [(p, *c, m) for p in REF_PAIRS if p != "USDJPY" for c in PAIR_CASES for m in TREND]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, args):
            rows += r
    # 10 ペアの表の USDJPY は、本体の同じ足・同じ版を使う
    lines = ["## 1. USDJPY: 方針と対策ごとの成績(順張り 8 本の取引の合計)", ""] + table_usdjpy(rows)
    lines += ["## 2. USDJPY: 手法ごと(期待値 / スワップを除く、pip / 取引。帰無との差の t、取引数)", ""]
    lines += table_methods(rows) + [""]
    lines += ["## 3. USDJPY: 方針の下での負けどころ"
              "(受取の基準。勝率 / 期待値 pip(スワップを除く)/ 負け額の割合 ÷ 取引の割合)", ""] + table_where(rows)
    lines += ["## 4. 主要 10 ペアのうち AUDJPY を除く 9 ペア(参考。1 万通貨の円、順張り 8 本の合計)", ""]
    lines += table_pairs(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
