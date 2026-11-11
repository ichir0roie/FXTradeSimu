"""実データだけでの全戦略の評価(E016、python -m fxsim pair-eval)。

合成の相場(並べ替え・偽薬・完全ランダム)を使わず、手元の実データを「通貨ペア x 期間(レンジ)」のパターンに切って、
動作ケースと方向でたらめ版(検証と同じ帰無)を対で回す。期間は足の長さで分ける(遡れる長さが違うので、レンジごとに長さも違う)。

    足(戦略)   実データ                              期間(レンジ)
    日足       相場の DB の日足 10 ペア(1993〜)        1996〜2026 年を 6 年ずつ 5 つ
    4 時間・1 時間  相場の DB の 1 時間足 10 ペア(2010〜)  2010-11〜2026-11 を 4 年ずつ 4 つ
    15 分・5 分    相場の DB の 5 分足 10 ペア(2025-07〜)   2025-07〜2026-11 を 4 か月ずつ 4 つ
    複数ペア      FRED の対円 7 ペアの日足(basket_real)    1999〜2026 年を 5〜6 年ずつ 5 つ

1 ペアの相場は real_pair(R012): 決済通貨を円に直し、記録のスプレッド、ペアの金利差のスワップ。

判定(結果を見る前に決めた基準。パターン = ペア x 期間):
    R1 利益            パターンの損益の中央値 > 0、プラスの割合 ≥ 60%、t ≥ 2
    R2 帰無より上       同じパターンでの差(モデル − 方向でたらめ版)の t ≥ 2
    R3 期間の安定       期間ごとの平均(全ペア)がプラスの期間が 75% 以上
    R4 ペアの広がり     ペアごとの平均(全期間)がプラスのペアが 60% 以上(複数ペアの戦略は対象外 = 通る)
    R5 スワップを除いて  スワップを除いた損益で R1 と同じ基準(金利差を取るだけの勝ちを落とす)
    合格 = 全部 / 有望 = R1・R2・R5 / それ以外 = 不合格
注意: 同じ期間のペアどうしは相関する(ドルや円の動きを共有する)ので、パターン単位の t は証拠の量を大きく見せる。
参考に「期間ごとの平均」の t(独立な証拠に近いが、数が 4〜5 しかない)も出す。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from .cases.model import BehaviorCase, TestCase
from .experiments.stats import summarize
from .storage import queries

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]


@dataclass(frozen=True)
class Range:
    data_tf: str                       # 相場の DB の時間足(D1 / H1 / M5)。複数ペアは "basket"
    windows: tuple[tuple[str, str], ...]


RANGES = {
    "D1": Range("D1", (("1996-01-01", "2001-12-31"), ("2002-01-01", "2007-12-31"), ("2008-01-01", "2013-12-31"),
                       ("2014-01-01", "2019-12-31"), ("2020-01-01", "2026-11-10"))),
    "H1": Range("H1", (("2010-11-01", "2014-10-31"), ("2014-11-01", "2018-10-31"), ("2018-11-01", "2022-10-31"),
                       ("2022-11-01", "2026-11-10"))),
    "M5": Range("M5", (("2025-07-14", "2025-10-31"), ("2025-11-01", "2026-02-28"), ("2026-03-01", "2026-06-30"),
                       ("2026-07-01", "2026-11-10"))),
    "basket": Range("basket", (("1999-01-01", "2004-12-31"), ("2005-01-01", "2010-12-31"),
                               ("2011-01-01", "2016-12-31"), ("2017-01-01", "2021-12-31"),
                               ("2022-01-01", "2026-11-10"))),
}


def range_for(b: BehaviorCase) -> Range:
    if b.is_multi():
        return RANGES["basket"]
    tf = int(b.strategy_params().get("timeframe") or 60)
    if tf >= 1440:
        return RANGES["D1"]
    if tf >= 60:
        return RANGES["H1"]
    return RANGES["M5"]


def pattern_label(pair: str, start: str) -> str:
    return f"{pair} {start[:7]}"


def broker_override(b: BehaviorCase, preset: str) -> dict:
    """動作ケースのブローカーを preset に差し替える。MT5 の口座の上書き(mt5)はそのまま残す。"""
    ov = (b.broker_spec.get("overrides") or {}).get("mt5")
    return {"preset": preset, "label": preset, "overrides": {"mt5": ov} if ov else {}}


def pair_eval_case(b: BehaviorCase, pairs: list[str] | None = None, broker: str | None = None,
                   daily_fill_hour: int | None = None) -> TestCase:
    """動作ケースを、ペア x 期間の実データのパターンで、方向でたらめ版と対にして回すテストケース。

    ID は P-<動作ケース>。"""
    r = range_for(b)
    patterns = []
    if r.data_tf == "basket":
        for start, end in r.windows:
            patterns.append({"market": "basket_real", "label": f"basket {start[:7]}",
                             "params": {"source": {"start": start, "end": end}}})
    else:
        for pair in pairs or PAIRS:
            for start, end in r.windows:
                src = {"path": f"db:{pair}/{r.data_tf}", "start": start, "end": end}
                if daily_fill_hour is not None and r.data_tf == "D1":
                    src["daily_fill_hour"] = int(daily_fill_hour)
                patterns.append({"market": "real_pair", "label": pattern_label(pair, start), "params": {"source": src}})
    years = max((np.datetime64(e) - np.datetime64(s)).astype(int) / 365.25 for s, e in r.windows)
    suffix = (f"-{broker}" if broker else "") + (f"-h{daily_fill_hour}" if daily_fill_hour is not None
                                                 and r.data_tf == "D1" else "")
    return TestCase(id=f"P-{b.id}{suffix}", behavior=b.id, patterns=patterns,
                    name=f"{b.id} の実データ評価(ペア x 期間、{r.data_tf}{'、' + broker if broker else ''})",
                    period={"years": round(float(years), 2)}, detail_seeds=[], null=True,
                    broker=broker_override(b, broker) if broker else None)


# ---- 判定 -----------------------------------------------------------------------

@dataclass
class PairCriteria:
    min_t: float = 2.0
    min_positive_ratio: float = 0.6
    min_positive_windows: float = 0.75
    min_positive_pairs: float = 0.6


def _split_label(label: str) -> tuple[str, str]:
    pair, _, win = label.partition(" ")
    return pair, win


def compute_pair_verdict(con: sqlite3.Connection, run_id: int, model_label: str,
                         criteria: PairCriteria | None = None) -> dict:
    c = criteria or PairCriteria()
    rows = []
    for d in queries.done_jobs_with_metrics(con, run_id):
        m = json.loads(d.pop("metrics_json") or "{}")
        costs = m.get("costs") or {}
        d["swap"] = float(costs.get("swap", m.get("swap_total", 0.0)) or 0.0)
        d["trades"] = int(m.get("n_trades", 0))
        d["years"] = float(m.get("years", 0.0) or 0.0)
        d["gross_pips"] = float(m.get("gross_pips_per_year", 0.0)) * d["years"]
        d["cost_pips"] = float(m.get("cost_pips_per_year", 0.0)) * d["years"]
        d["pair"], d["window"] = _split_label(d["market_label"])
        rows.append(d)
    model = [r for r in rows if r["strategy_label"] == model_label]
    null = {r["path_id"]: r for r in rows if r["strategy_label"] != model_label}
    pnl = np.array([r["pnl"] for r in model], dtype=float)
    ex_swap = np.array([r["pnl"] - r["swap"] for r in model], dtype=float)
    paired = np.array([r["pnl"] - null[r["path_id"]]["pnl"] for r in model if r["path_id"] in null], dtype=float)
    s_all, s_ex, s_pair = summarize(pnl), summarize(ex_swap), summarize(paired)

    def group_means(key):
        out: dict[str, list[float]] = {}
        for r in model:
            out.setdefault(r[key], []).append(r["pnl"])
        return {k: float(np.mean(v)) for k, v in sorted(out.items())}

    by_window, by_pair = group_means("window"), group_means("pair")
    multi = all(p == "basket" for p in by_pair)
    pos_windows = float(np.mean([v > 0 for v in by_window.values()])) if by_window else 0.0
    pos_pairs = float(np.mean([v > 0 for v in by_pair.values()])) if by_pair else 0.0
    s_win = summarize(list(by_window.values()))

    def profit_ok(s):
        return bool(s.get("n")) and s["median"] > 0 and s["positive_ratio"] >= c.min_positive_ratio \
            and s["t"] >= c.min_t

    checks = [
        {"key": "R1", "label": "利益", "ok": profit_ok(s_all), "detail": _fmt(s_all)},
        {"key": "R2", "label": "方向でたらめ版より上", "ok": bool(s_pair.get("n")) and s_pair["t"] >= c.min_t,
         "detail": f"差の平均 {s_pair.get('mean', 0):,.0f} 円 / t {s_pair.get('t', 0):.2f}(n={s_pair.get('n', 0)})"},
        {"key": "R3", "label": "期間の安定", "ok": pos_windows >= c.min_positive_windows,
         "detail": f"平均がプラスの期間 {pos_windows:.0%}({len(by_window)} 期間)/ "
                   f"期間の平均の t {s_win.get('t', 0):.2f}"},
        {"key": "R4", "label": "ペアの広がり", "ok": multi or pos_pairs >= c.min_positive_pairs,
         "detail": "複数ペアの戦略(対象外)" if multi else f"平均がプラスのペア {pos_pairs:.0%}({len(by_pair)} ペア)"},
        {"key": "R5", "label": "スワップを除いて", "ok": profit_ok(s_ex), "detail": _fmt(s_ex)},
    ]
    ok = {ch["key"]: ch["ok"] for ch in checks}
    if all(ok.values()):
        verdict = "合格"
    elif ok["R1"] and ok["R2"] and ok["R5"]:
        verdict = "有望"
    else:
        verdict = "不合格"
    n_trades = sum(r["trades"] for r in model)
    n_null = sum(r["trades"] for r in null.values())
    years = sum(r["years"] for r in model)
    return {
        "verdict": verdict, "checks": checks, "model": model_label,
        "pnl": s_all, "pnl_ex_swap": s_ex, "paired": s_pair, "by_window": by_window, "by_pair": by_pair,
        "window_t": s_win.get("t", 0.0),
        "trades_per_year": n_trades / years if years else 0.0,
        "gross_pips_per_trade": sum(r["gross_pips"] for r in model) / n_trades if n_trades else 0.0,
        "cost_pips_per_trade": sum(r["cost_pips"] for r in model) / n_trades if n_trades else 0.0,
        "null_gross_pips_per_trade": (sum(r["gross_pips"] for r in null.values()) / n_null) if n_null else 0.0,
        "swap_mean": float(np.mean([r["swap"] for r in model])) if model else 0.0,
        "ruin_ratio": float(np.mean([bool(r["ruined"]) for r in model])) if model else 0.0,
        "patterns": [{"label": r["market_label"], "pnl": r["pnl"], "swap": r["swap"], "trades": r["trades"],
                      "null_pnl": null[r["path_id"]]["pnl"] if r["path_id"] in null else None} for r in model],
    }


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "パターンがありません"
    return (f"中央値 {s['median']:,.0f} 円 / 平均 {s['mean']:,.0f} 円 / プラス {s['positive_ratio']:.0%} / "
            f"t {s['t']:.2f}(n={s['n']})")


def format_pair_verdict(v: dict) -> str:
    lines = [f"### {v['model']}: {v['verdict']}", "", "| 条件 | 結果 | 根拠 |", "|---|---|---|"]
    for ch in v["checks"]:
        lines.append(f"| {ch['key']} {ch['label']} | {'通過' if ch['ok'] else '不通過'} | {ch['detail']} |")
    lines += ["", f"取引 / 年 {v['trades_per_year']:.1f}、素の損益 / 取引 {v['gross_pips_per_trade']:+.2f} pip"
                  f"(方向でたらめ版 {v['null_gross_pips_per_trade']:+.2f})、"
                  f"費用 / 取引 {v['cost_pips_per_trade']:.2f} pip、"
                  f"スワップの平均 {v['swap_mean']:,.0f} 円、破綻 {v['ruin_ratio']:.0%}",
              "", "期間ごとの平均(円): " + ", ".join(f"{k} {x:+,.0f}" for k, x in v["by_window"].items()),
              "", "ペアごとの平均(円): " + ", ".join(f"{k} {x:+,.0f}" for k, x in v["by_pair"].items())]
    return "\n".join(lines)


__all__ = ["PAIRS", "RANGES", "PairCriteria", "compute_pair_verdict", "format_pair_verdict", "pair_eval_case",
           "range_for"]
