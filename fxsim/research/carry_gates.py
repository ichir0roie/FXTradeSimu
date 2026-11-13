"""E038: 本命のスワップ狙い(B049)の負けどころを確かめ、新規の関門で潰す。

    python -m fxsim.research.carry_gates [--workers N]

相場は 30 年の並べ替え(real_pairs_bootstrap、ブロック 250 日、約定は 2 時)の seed 101〜120。verify が使う
実データと seed 1〜10 は、関門を選んだ後の本番の検証に取っておく(同じ相場で選んで合否を出さない)。
1. B049 の取引に、入る判断をした足の局面を付けて負けの集まり方を見る(どれも因果的):
    荒さ     ATR(20) / 直前 500 本の ATR(20) の中央値: 静か < 0.8 / 中 / 荒い > 1.25(E036 の閾値)
    曜日     約定した取引日(22:00 UTC で切り替わる): 月 / 火〜木 / 金
    金利差   受け取る年率(業者の取り分を引いた後): 2〜3% / 3〜5% / 5% 超
    一方向   20 本の効率比: もみ合い < 0.2 / 中 / 一方向 > 0.4
    流れ     60 本の対数リターン / (20 本のボラ x √60) を建てた向きで: 逆行 < −0.5 / 中立 / 順行 > +0.5
    同時     入った足で口座が持っていたペアの数: 0〜1 / 2〜3 / 4 以上
2. 関門(派生の戦略 main_carry_gate、S043)を掛けた版と B049 を、同じ相場で比べる。選ぶ基準は E038 の提案書
"""
from __future__ import annotations

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.calendar import trading_day
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa: F403
from fxsim.experiments.config import default_workers
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES

SPAN = ("1996-11-11", "2026-11-10")
SEEDS = list(range(101, 121))
NS_DAY = 86_400_000_000_000
FEATS = {
    "荒さ": ("静か", "中", "荒い"),
    "曜日": ("月", "火〜木", "金"),
    "金利差": ("2〜3%", "3〜5%", "5% 超"),
    "一方向": ("もみ合い", "中", "一方向"),
    "流れ": ("逆行", "中立", "順行"),
    "同時": ("0〜1", "2〜3", "4 以上"),
}
# (ラベル, 戦略, 追加の引数)。gate の引数は main_carry_gate のもの
VARIANTS = [
    ("B049", "main_carry_breakout", {}),
    ("G1 荒いときは新規なし", "main_carry_gate", {"entry_vol_max": 1.25}),
    ("G2 流れに逆らう抜けは新規なし", "main_carry_gate", {"counter_z": 0.5}),
    ("G4 金利差の下限 3%", "main_carry_breakout", {"min_carry": 0.03}),
    ("G5 同時 3 ペアまで", "main_carry_gate", {"max_held": 3}),
    ("G1+G2", "main_carry_gate", {"entry_vol_max": 1.25, "counter_z": 0.5}),
    ("G1+G2+G5", "main_carry_gate", {"entry_vol_max": 1.25, "counter_z": 0.5, "max_held": 3}),
]


def market(kind: str, seed: int):
    src = {"start": SPAN[0], "end": SPAN[1], "daily_fill_hour": 2}
    if kind == "real":
        return MARKETS.create("real_pairs", {"source": src}).generate(seed=1)
    return MARKETS.create("real_pairs_bootstrap", {"source": {**src, "block_days": 250}}).generate(seed=seed)


def run(md, strategy):
    b = load_cases().behaviors["B049"]
    bt = build_backtester(md, strategy, BrokerConfig.preset("oanda"), EngineConfig(**b.engine), seed=1)
    bt.run()
    return bt


def _cut(v, lo, hi, names):
    if not np.isfinite(v):
        return None
    return names[0] if v < lo else names[2] if v > hi else names[1]


def features(s, pair: str):
    c, a = s.c[pair], s.atr[pair]
    n = len(c)
    vr = np.full(n, np.nan)
    for i in range(520, n):
        m = np.nanmedian(a[i - 500:i])
        vr[i] = a[i] / m if m > 0 else np.nan
    lc = np.log(c)
    r = np.diff(lc, prepend=lc[0])
    cs = np.cumsum(np.abs(r))
    er = np.full(n, np.nan)
    er[20:] = np.abs(lc[20:] - lc[:-20]) / np.maximum(cs[20:] - cs[:-20], 1e-12)
    vol20 = np.full(n, np.nan)
    for i in range(20, n):
        vol20[i] = r[i - 19:i + 1].std()
    trend = np.full(n, np.nan)
    trend[60:] = (lc[60:] - lc[:-60]) / np.maximum(vol20[60:] * np.sqrt(60), 1e-12)
    return vr, er, trend


def tagged_trades(bt, s) -> list[dict]:
    feats = {pair: features(s, pair) for pair in s.pairs}
    opens = [(t.entry_t, t.exit_t) for t in bt.trades]
    out = []
    for t in bt.trades:
        if t.reason == "end":
            continue
        k = int(np.searchsorted(s.t, t.entry_t, "right")) - 2
        if k < 0:
            continue
        vr, er, trend = feats[t.pair]
        dow = int((trading_day(t.entry_t) + 3) % 7)        # 22:00 UTC で切り替わる取引日の曜日
        carry = float(s.carry[t.pair][k])
        held_now = sum(1 for a, b in opens if a < t.entry_t <= b)
        out.append({
            "pnl": float(t.pnl), "swap": float(t.swap), "pair": t.pair,
            "days": (t.exit_t - t.entry_t) / NS_DAY,
            "荒さ": _cut(vr[k], 0.8, 1.25, FEATS["荒さ"]),
            "曜日": "月" if dow == 0 else "金" if dow == 4 else "火〜木" if dow in (1, 2, 3) else None,
            "金利差": "2〜3%" if carry < 0.03 else "3〜5%" if carry <= 0.05 else "5% 超",
            "一方向": _cut(er[k], 0.2, 0.4, FEATS["一方向"]),
            "流れ": _cut(trend[k] * t.side, -0.5, 0.5, FEATS["流れ"]),
            "同時": "0〜1" if held_now <= 1 else "2〜3" if held_now <= 3 else "4 以上",
        })
    return out


def summary(bt) -> dict:
    day_t, eq = daily_equity(bt)
    v = np.asarray(eq, dtype=float)
    r = np.diff(v) / v[:-1]
    peak = np.maximum.accumulate(v)
    years = (day_t[-1] - day_t[0]) / NS_DAY / 365.25
    return {"final": float(v[-1]), "cagr": float((v[-1] / v[0]) ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(260)) if r.std() > 0 else 0.0,
            "mdd": float(((peak - v) / peak).max()), "trades": len(bt.trades),
            "swap": float(sum(t.swap for t in bt.trades))}


def job(spec):
    kind, seed, label, strat, params = spec
    with contextlib.suppress(OSError):
        os.nice(10)
    md = market(kind, seed)
    b = load_cases().behaviors["B049"]
    s = STRATEGIES.create(strat, {**b.strategy_params(), **params})
    bt = run(md, s)
    return {"kind": kind, "seed": seed, "label": label, "summary": summary(bt),
            "trades": tagged_trades(bt, s) if label == "B049" else []}


def table_where(rows) -> list[str]:
    out = []
    for kind, name in (("boot", f"並べ替え {len(SEEDS)} 本(seed {SEEDS[0]}〜{SEEDS[-1]})"),):
        tr = [t for r in rows if r["kind"] == kind and r["label"] == "B049" for t in r["trades"]]
        loss_total = -sum(t["pnl"] for t in tr if t["pnl"] < 0)
        out += [f"### {name}({len(tr)} 取引、勝率 {np.mean([t['pnl'] > 0 for t in tr]):.0%}、"
                f"1 取引 {np.mean([t['pnl'] for t in tr]):+,.0f} 円)", "",
                "| 切り口 | 区分 1 | 区分 2 | 区分 3 |", "|---|---:|---:|---:|"]
        for feat, cats in FEATS.items():
            cells = []
            for c in cats:
                sub = [t for t in tr if t[feat] == c]
                if len(sub) < 10:
                    cells.append(f"{c}: -({len(sub)})")
                    continue
                share = -sum(t["pnl"] for t in sub if t["pnl"] < 0) / max(loss_total, 1e-9) / (len(sub) / len(tr))
                ex = np.mean([t["pnl"] - t["swap"] for t in sub])
                win, mean = np.mean([t["pnl"] > 0 for t in sub]), np.mean([t["pnl"] for t in sub])
                cells.append(f"{c}: {win:.0%} / {mean:+,.0f}(除 {ex:+,.0f})/ 負け {share:.2f}x / {len(sub)}")
            out.append(f"| {feat} | " + " | ".join(cells) + " |")
        hold = [("〜5 日", lambda d: d <= 5), ("6〜20 日", lambda d: 5 < d <= 20), ("21 日〜", lambda d: d > 20)]
        cells = []
        for name_h, f in hold:
            sub = [t for t in tr if f(t["days"])]
            if not sub:
                cells.append(f"{name_h}: -")
                continue
            share = -sum(t["pnl"] for t in sub if t["pnl"] < 0) / max(loss_total, 1e-9) / (len(sub) / len(tr))
            win, mean = np.mean([t["pnl"] > 0 for t in sub]), np.mean([t["pnl"] for t in sub])
            cells.append(f"{name_h}: {win:.0%} / {mean:+,.0f} / 負け {share:.2f}x / {len(sub)}")
        out += ["| 持った日数 | " + " | ".join(cells) + " |", ""]
    return out


def table_variants(rows) -> list[str]:
    out = ["| 版 | 最終資産の中央値(万円) | 年率の中央値 | シャープの平均 | B049 よりシャープが高い本数 | "
           "最大 DD の中央値 / 最悪 | 取引の平均 | スワップの平均(万円) | 基準 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    base = {r["seed"]: r["summary"] for r in rows if r["kind"] == "boot" and r["label"] == "B049"}
    base_sh = np.mean([v["sharpe"] for v in base.values()])
    base_dd = max(v["mdd"] for v in base.values())
    for label, *_ in VARIANTS:
        boot = {r["seed"]: r["summary"] for r in rows if r["kind"] == "boot" and r["label"] == label}
        better = sum(boot[k]["sharpe"] > base[k]["sharpe"] for k in boot)
        sh = np.mean([v["sharpe"] for v in boot.values()])
        dd = [v["mdd"] for v in boot.values()]
        ok = label != "B049" and sh > base_sh and better >= 14 and max(dd) <= base_dd
        out.append(f"| {label} | {np.median([v['final'] for v in boot.values()]) / 1e4:,.1f} | "
                   f"{np.median([v['cagr'] for v in boot.values()]):+.1%} | {sh:.3f} | {better}/{len(boot)} | "
                   f"{np.median(dd):.0%} / {max(dd):.0%} | {np.mean([v['trades'] for v in boot.values()]):.0f} | "
                   f"{np.mean([v['swap'] for v in boot.values()]) / 1e4:+.1f} | {'通る' if ok else '-'} |")
    return out


# E039: 儲けに対するドローダウン(カルマー比 = 年率 ÷ 最大 DD)。量で動かない物差しで、関門が量を減らすより良いかを見る
DD_SEEDS = list(range(121, 141))
DD_VARIANTS = [VARIANTS[0], VARIANTS[1], VARIANTS[2], VARIANTS[5]]       # B049・G1・G2・G1+G2(閾値は E038 のまま)


def calmar(v: dict) -> float:
    return v["cagr"] / v["mdd"] if v["mdd"] > 0 else float("nan")


def table_dd(rows) -> list[str]:
    out = ["| 版 | カルマー比の平均 / 中央値 | B049 より上の本数 | 年率の中央値 | 最大 DD の中央値 / 最悪 | "
           "シャープの平均 | 取引の平均 | 基準 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    base = {r["seed"]: r["summary"] for r in rows if r["label"] == "B049"}
    base_cal = np.mean([calmar(v) for v in base.values()])
    base_dd = max(v["mdd"] for v in base.values())
    base_sh = np.mean([v["sharpe"] for v in base.values()])
    for label, *_ in DD_VARIANTS:
        boot = {r["seed"]: r["summary"] for r in rows if r["label"] == label}
        cal = [calmar(v) for v in boot.values()]
        better = sum(calmar(boot[k]) > calmar(base[k]) for k in boot)
        dd = [v["mdd"] for v in boot.values()]
        sh = np.mean([v["sharpe"] for v in boot.values()])
        ok = (label != "B049" and np.mean(cal) > base_cal and better >= 14 and max(dd) <= base_dd - 0.05
              and sh >= base_sh - 0.02)
        out.append(f"| {label} | {np.mean(cal):.3f} / {np.median(cal):.3f} | {better}/{len(boot)} | "
                   f"{np.median([v['cagr'] for v in boot.values()]):+.1%} | {np.median(dd):.0%} / {max(dd):.0%} | "
                   f"{sh:.3f} | {np.mean([v['trades'] for v in boot.values()]):.0f} | {'通る' if ok else '-'} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--dd", action="store_true", help="E039: seed 121〜140 でカルマー比を比べる")
    a = ap.parse_args()
    if a.dd:
        specs = [("boot", x, *v) for v in DD_VARIANTS for x in DD_SEEDS]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(job, specs))
        print("\n".join([f"## カルマー比(年率 ÷ 最大 DD)、並べ替え seed {DD_SEEDS[0]}〜{DD_SEEDS[-1]}", "",
                         "基準: カルマー比の平均 > B049、14/20 本以上で B049 より上、"
                         "最大 DD の最悪が B049 − 5 ポイント以下、シャープの平均 ≥ B049 − 0.02", ""] + table_dd(rows)))
        return
    specs = [("boot", x, *v) for v in VARIANTS for x in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, specs))
    lines = ["## 1. B049 の負けどころ(勝率 / 1 取引の円(スワップを除く)/ 負け額の割合 ÷ 取引の割合 / 取引数)", ""]
    lines += table_where(rows)
    lines += ["## 2. 関門を掛けた版(同じ相場。基準: シャープの平均 > B049、14/20 本以上で B049 より上、"
              "最大 DD の最悪が B049 以下)", ""] + table_variants(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
