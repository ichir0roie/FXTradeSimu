"""ひとつのモデル(動作ケース)が長期的に利益を出すかの検証。

同じ期間・同じシードの相場パターンの組で、モデルと「方向だけでたらめ版」(direction_null)を対にして回し、
次の条件で合否を判定する。数字はすべて記録され、画面(/cases/<テストケース>)で見られる。

    C1 利益        実データ系のパターンで、損益の中央値 > 0、プラスの割合 ≥ 60%、パターン単位の t ≥ 2
    C2 帰無より上  同じパターンどうしの差(モデル − 方向でたらめ)の t ≥ 2
    C3 偽薬より上  実データ系の損益が、偽薬(符号でたらめの骨格)の損益より大きい(ウェルチの t ≥ 2)
    C4 時間の安定性 年ごとの損益の中央値がプラスの年が 60% 以上、かつ前半・後半の平均がどちらもプラス
    参考           完全ランダム相場(stylized / malicious)とストレス(real_bridge_malicious)の損益

判定: C1〜C4 すべて → 合格 / C1 と C2 → 条件付き / それ以外 → 不合格
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from .experiments.stats import summarize
from .storage import queries

NULL_LABEL = "NULL"

# 市場モデル -> パターンの組
GROUPS = {
    "real": "real", "real_bridge": "real", "real_bootstrap": "real",
    "real_bridge_placebo": "placebo",
    "stylized": "synthetic", "malicious": "synthetic", "random_walk": "synthetic",
    "real_bridge_malicious": "stress",
}
GROUP_LABELS = {"real": "実データ系", "placebo": "偽薬", "synthetic": "完全ランダム", "stress": "ストレス"}

# 検証の標準のパターン(seeds は実行時に入る。real は決定的なので 1 本)
STANDARD_PATTERNS = ["real", "real_bridge", "real_bootstrap", "real_bridge_placebo", "stylized", "malicious",
                     "real_bridge_malicious"]


@dataclass
class Criteria:
    min_t: float = 2.0
    min_positive_ratio: float = 0.6
    min_positive_years: float = 0.6


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def _rows(con: sqlite3.Connection, run_id: int) -> list[dict]:
    out = []
    for d in queries.done_jobs_with_metrics(con, run_id):
        m = json.loads(d.pop("metrics_json") or "{}")
        d["yearly"] = m.get("yearly_pnl", {})
        d["group"] = GROUPS.get(d["market"], "synthetic")
        out.append(d)
    return out


def _year_coverage(rows: list[dict]) -> dict[str, float]:
    """各暦年のうち、パターンの期間が覆っている割合(実データ系のパターンの最小の開始〜最大の終わり)。"""
    if not rows:
        return {}
    t0 = np.datetime64(min(r["t_start"] for r in rows), "s")
    t1 = np.datetime64(max(r["t_end"] for r in rows), "s")
    out = {}
    for y in range(int(str(t0)[:4]), int(str(t1)[:4]) + 1):
        a = max(t0, np.datetime64(f"{y}-01-01", "s"))
        b = min(t1, np.datetime64(f"{y + 1}-01-01", "s"))
        out[str(y)] = float((b - a) / np.timedelta64(365 * 86400, "s"))
    return out


def compute_verdict(con: sqlite3.Connection, run_id: int, model_label: str,
                    criteria: Criteria | None = None) -> dict:
    """run の結果から判定を作る。model_label は戦略のラベル(動作ケースの ID)。"""
    c = criteria or Criteria()
    rows = _rows(con, run_id)
    model = [r for r in rows if r["strategy_label"] == model_label]
    null = {r["path_id"]: r for r in rows if r["strategy_label"] == NULL_LABEL}

    def pnl(group):
        return np.array([r["pnl"] for r in model if r["group"] == group], dtype=float)

    real, placebo = pnl("real"), pnl("placebo")
    s_real = summarize(real)
    paired = np.array([r["pnl"] - null[r["path_id"]]["pnl"] for r in model
                       if r["group"] == "real" and r["path_id"] in null], dtype=float)
    s_paired = summarize(paired)

    # 年ごと(実データ系の全パターン)。期間が半年に満たない端の年は安定性の判定に使わない
    real_rows = [r for r in model if r["group"] == "real"]
    years = sorted({y for r in real_rows for y in r["yearly"]})
    covered = _year_coverage(real_rows)
    yearly = []
    for y in years:
        vals = [r["yearly"].get(y) for r in model if r["group"] == "real" and y in r["yearly"]]
        s = summarize(vals)
        if s:
            yearly.append({"year": y, "coverage": covered.get(y, 0.0), **s})
    full_years = [y for y in yearly if y["coverage"] >= 0.5]
    pos_years = sum(1 for y in full_years if y["median"] > 0) / len(full_years) if full_years else 0.0
    half = len(full_years) // 2
    first = float(np.mean([y["mean"] for y in full_years[:half]])) if half else 0.0
    second = float(np.mean([y["mean"] for y in full_years[half:]])) if full_years[half:] else 0.0

    checks = []

    def check(key, label, ok, detail):
        checks.append({"key": key, "label": label, "ok": bool(ok), "detail": detail})

    if s_real:
        check("C1", "利益(実データ系)",
              s_real["median"] > 0 and s_real["positive_ratio"] >= c.min_positive_ratio and s_real["t"] >= c.min_t,
              f"中央値 {s_real['median']:,.0f} 円 / プラス {s_real['positive_ratio']:.0%} / t {s_real['t']:.2f}"
              f"(n={s_real['n']})")
    else:
        check("C1", "利益(実データ系)", False, "実データ系のパターンがありません")
    if s_paired:
        check("C2", "方向でたらめ版より上", s_paired["t"] >= c.min_t,
              f"差の平均 {s_paired['mean']:,.0f} 円 / t {s_paired['t']:.2f}(n={s_paired['n']})")
    else:
        check("C2", "方向でたらめ版より上", False, "対になる帰無の結果がありません")
    t_pl = _welch_t(real, placebo)
    if len(placebo):
        check("C3", "偽薬より上", t_pl >= c.min_t,
              f"実データ系 {real.mean():,.0f} 円 − 偽薬 {placebo.mean():,.0f} 円 / t {t_pl:.2f}")
    else:
        check("C3", "偽薬より上", False, "偽薬のパターンがありません")
    check("C4", "時間の安定性", pos_years >= c.min_positive_years and first > 0 and second > 0,
          f"中央値がプラスの年 {pos_years:.0%}({len(full_years)} 年中)/ 前半 {first:,.0f} 円 / 後半 {second:,.0f} 円")

    ok = {ch["key"]: ch["ok"] for ch in checks}
    if all(ok.values()):
        verdict = "合格"
    elif ok["C1"] and ok["C2"]:
        verdict = "条件付き"
    else:
        verdict = "不合格"

    groups = {}
    for g in GROUP_LABELS:
        vals = pnl(g)
        if len(vals):
            groups[g] = {"label": GROUP_LABELS[g], **summarize(vals)}
            nulls = [null[r["path_id"]]["pnl"] for r in model if r["group"] == g and r["path_id"] in null]
            if nulls:
                groups[g]["null_mean"] = float(np.mean(nulls))
    return {
        "verdict": verdict,
        "checks": checks,
        "groups": groups,
        "paired": s_paired,
        "yearly": yearly,
        "criteria": c.__dict__,
        "model": model_label,
    }


def format_verdict(v: dict) -> str:
    lines = [f"## 判定: {v['verdict']}({v['model']})", ""]
    for ch in v["checks"]:
        lines.append(f"- {'✓' if ch['ok'] else '✗'} {ch['key']} {ch['label']}: {ch['detail']}")
    lines += ["", "| パターンの組 | n | 損益 平均 | 中央 | プラス | t | 方向でたらめ版の平均 |",
              "|---|---|---|---|---|---|---|"]
    for g in v["groups"].values():
        lines.append(f"| {g['label']} | {g['n']} | {g['mean']:,.0f} | {g['median']:,.0f} | {g['positive_ratio']:.0%} | "
                     f"{g['t']:.2f} | {g.get('null_mean', float('nan')):,.0f} |")
    if v["yearly"]:
        lines += ["", "年ごと(実データ系): " + ", ".join(f"{y['year']} {y['median']:+,.0f}" for y in v["yearly"])]
    return "\n".join(lines)
