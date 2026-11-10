"""パターン(シード・実データ)ごとの結果をまとめる統計。

同じ戦略を、実データと複数のランダム相場で回した結果を並べ、
平均・合計(総合)・中央値・標準偏差・最小/最大・10/90 パーセンタイル・勝ちパターンの割合・t 値を出す。
取引単位の統計(素のシャープなど)は同じ経路の中で取引が独立でないため有意さを過大に見せる。
当否の判定はパターン単位の t 値で行う。
"""
from __future__ import annotations

import sqlite3

import numpy as np

STAT_METRICS = ("pnl", "net_pips_per_year", "sharpe", "max_dd", "gross_sharpe", "trades_per_year", "win_rate")


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "total": float(v.sum()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "positive_ratio": float((v > 0).mean()),
        # パターン単位の t 値。パターンどうしは独立なので、有意かどうかはこれで見る(|t| > 2 が目安)
        "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std(ddof=1) > 0 else 0.0,
    }


def job_rows(con: sqlite3.Connection, run_id: int) -> list[dict]:
    q = """SELECT j.id, p.market_label, p.market, p.seed, j.broker, j.strategy_label, j.status, j.note,
                  j.pnl, j.net_pips_per_year, j.sharpe, j.max_dd, j.gross_sharpe, j.trades_per_year, j.win_rate,
                  j.cost_pips_per_year, j.gross_pips_per_year, j.final_equity, j.ruined
           FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ? ORDER BY p.market_label, p.seed"""
    return [dict(r) for r in con.execute(q, (run_id,))]


def strategy_stats(con: sqlite3.Connection, run_id: int, metrics=STAT_METRICS) -> list[dict]:
    """(ブローカー, 戦略, 市場) ごと + 市場をまとめた "ALL" 行。各指標に summarize を掛ける。"""
    rows = [r for r in job_rows(con, run_id) if r["status"] == "done"]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["broker"], r["strategy_label"], r["market_label"]), []).append(r)
        groups.setdefault((r["broker"], r["strategy_label"], "ALL"), []).append(r)
    out = []
    for (broker, strat, market), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] == "ALL", kv[0][2])):
        item = {"broker": broker, "strategy": strat, "market": market, "n": len(rs),
                "patterns": [{"market": r["market_label"], "seed": r["seed"], "pnl": r["pnl"],
                              "sharpe": r["sharpe"]} for r in rs]}
        for m in metrics:
            item[m] = summarize([r[m] for r in rs])
        out.append(item)
    return out
