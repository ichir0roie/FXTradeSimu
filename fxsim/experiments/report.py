"""実験結果を Markdown の表にする(CLI の report / Claude が結果を読むための出力)。"""
from __future__ import annotations

import sqlite3

from .stats import strategy_stats


def _f(x, nd=2, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{x * 100:.{nd}f}%"
    if abs(x) >= 10000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_report(con: sqlite3.Connection, run_id: int, patterns: bool = False) -> str:
    run = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        return f"run #{run_id} not found"
    lines = [f"# run #{run_id} {run['name']}",
             f"- 状態: {run['status']} / 作成: {run['created_at']} / 所要: {_f(run['elapsed_sec'], 0)}s / git: {run['git_sha']}",
             f"- 期間: {run['years']} 年 / シード: {run['seeds_json']} / 提案: {run['proposal'] or '-'}", ""]
    ms = con.execute("SELECT * FROM v_market_summary WHERE run_id=? ORDER BY market_label", (run_id,)).fetchall()
    if ms:
        lines += ["## 市場の性質(シード平均)", _table(
            ["市場", "n", "現実らしさ", "外れ数", "年率ボラ", "尖度H1", "|r|自己相関", "分散比H1→D1", "ダマシ率", "ヒゲ戻り率"],
            [[m["market_label"], m["n_seeds"], _f(m["realism_score"], 1), _f(m["n_fail"], 1),
              _f(m["annual_vol"], 3), _f(m["kurt_h1"], 1), _f(m["acf_abs_h1_lag1"], 3), _f(m["vr_h1_d1"], 3),
              _f(m["false_breakout_rate"], 3), _f(m["wick_rejection_rate"], 3)] for m in ms]), ""]
    stats = strategy_stats(con, run_id)
    if stats:
        lines += ["## 戦略の損益(パターンごとの統計。ALL は全市場をまとめたもの)",
                  "損益は円(初期 100 万円・1 万通貨固定が既定)。勝率はパターン(シード)単位で損益がプラスだった割合。", ""]
        rows = []
        for s in stats:
            p = s["pnl"]
            if not p.get("n"):
                continue
            rows.append([s["broker"], s["strategy"], s["market"], p["n"], _f(p["mean"], 0), _f(p["median"], 0),
                         _f(p["total"], 0), _f(p["std"], 0), _f(p["min"], 0), _f(p["max"], 0),
                         _f(p["positive_ratio"], 0, pct=True), _f(p["t"]), _f(s["sharpe"]["mean"]),
                         _f(s["gross_sharpe"]["mean"]), _f(s["max_dd"]["mean"], 1, pct=True),
                         _f(s["trades_per_year"]["mean"], 0)])
        lines.append(_table(["ブローカー", "戦略", "市場", "n", "損益 平均", "中央", "合計", "標準偏差", "最小", "最大",
                             "勝ちパターン", "t値", "シャープ", "素のシャープ", "最大DD", "取引/年"], rows))
        lines.append("")
        if patterns:
            lines.append("## パターンごとの損益")
            for s in stats:
                if s["market"] == "ALL":
                    continue
                pats = ", ".join(f"seed{p['seed']}={_f(p['pnl'], 0)}" for p in s["patterns"])
                lines.append(f"- {s['broker']} / {s['strategy']} / {s['market']}: {pats}")
    skipped = con.execute("SELECT strategy_label, note, COUNT(*) n FROM jobs WHERE run_id=? AND status='skipped'"
                          " GROUP BY strategy_label, note", (run_id,)).fetchall()
    if skipped:
        lines += ["", "## 回さなかったジョブ"] + [f"- {r['strategy_label']}: {r['note']} ({r['n']} 本)" for r in skipped]
    return "\n".join(lines)


def leaderboard(con: sqlite3.Connection, kind: str = "strategies", limit: int = 30) -> str:
    if kind == "markets":
        rows = con.execute("SELECT * FROM v_market_summary ORDER BY run_id DESC, market_label LIMIT ?",
                           (limit,)).fetchall()
        return _table(["run", "市場", "版", "params", "n", "現実らしさ", "年率ボラ", "尖度H1", "分散比", "ダマシ率"],
                      [[r["run_id"], r["market_label"], r["market_version"], r["params_hash"], r["n_seeds"],
                        _f(r["realism_score"], 1), _f(r["annual_vol"], 3), _f(r["kurt_h1"], 1),
                        _f(r["vr_h1_d1"], 3), _f(r["false_breakout_rate"], 3)] for r in rows])
    rows = con.execute("SELECT * FROM v_strategy_summary ORDER BY sharpe DESC LIMIT ?", (limit,)).fetchall()
    return _table(["run", "市場", "ブローカー", "戦略", "n", "シャープ", "素のシャープ", "損益平均", "pip/年", "最大DD"],
                  [[r["run_id"], r["market_label"], r["broker"], r["strategy_label"], r["n_seeds"], _f(r["sharpe"]),
                    _f(r["gross_sharpe"]), _f(r["pnl_mean"], 0), _f(r["net_pips_per_year"], 0),
                    _f(r["max_dd"], 1, pct=True)] for r in rows])


__all__ = ["run_report", "leaderboard"]
