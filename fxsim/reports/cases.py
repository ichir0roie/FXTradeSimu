"""動作ケース・テストケースの一覧と最新結果の Markdown。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..cases import load_cases
from ..storage import queries
from ..strategies import STRATEGIES
from .markdown import table


def cases_overview(con: sqlite3.Connection, root: Path | str | None = None) -> str:
    book = load_cases(root)
    rows = []
    for b in book.behaviors.values():
        s = b.trading_schedule()
        hours = (", ".join(f"{x}-{y}" for x, y in s.hours) + f" {s.tz}") if s.hours else "終日"
        tf = b.strategy_params().get("timeframe") or STRATEGIES.create(b.strategy).timeframe
        rows.append([b.id, b.name, b.strategy, f"{tf}m", f"{s.decision_every} 本ごと", f"{hours} ({','.join(s.days)})",
                     s.outside, b.broker_spec["preset"]])
    lines = ["## 動作ケース",
             table(["ID", "名前", "戦略", "足", "判断の間隔", "取引時間", "時間外", "ブローカー"], rows), ""]
    rows = []
    for t in book.tests.values():
        pats = ", ".join(f"{p.display_label}x{len(p.seeds)}" for p in t.patterns)
        latest = queries.latest_case_run(con, t.id)
        p = ((latest[1].get("all") or {}).get("pnl") or {}) if latest else {}
        if latest and p.get("n"):
            cells = [f"#{latest[0]}", f"{p['mean']:,.0f}", f"{p['median']:,.0f}", f"{p['total']:,.0f}",
                     f"{p['positive_ratio'] * 100:.0f}%", f"{p.get('t', 0):.2f}",
                     (latest[1].get("verdict") or {}).get("verdict", "-")]
        else:
            cells = ["未実行", "-", "-", "-", "-", "-", "-"]
        rows.append([t.id, t.name, t.behavior, t.period_label(), pats, *cells])
    for tid in queries.case_test_ids(con):         # fxsim verify が作った検証(YAML には無い)
        if tid in book.tests:
            continue
        run_id, summary = queries.latest_case_run(con, tid) or (None, {})
        p = ((summary.get("all") or {}).get("pnl") or {})
        if not p.get("n"):
            continue
        rows.append([tid, "(fxsim verify)", tid.removeprefix("V-"), "-", "標準パターン", f"#{run_id}",
                     f"{p['mean']:,.0f}", f"{p['median']:,.0f}", f"{p['total']:,.0f}",
                     f"{p['positive_ratio'] * 100:.0f}%", f"{p.get('t', 0):.2f}",
                     (summary.get("verdict") or {}).get("verdict", "-")])
    headers = ["ID", "名前", "動作", "期間", "パターン", "最新 run", "損益 平均", "中央", "合計", "プラス", "t値",
               "判定"]
    lines += ["## テストケース", table(headers, rows)]
    return "\n".join(lines)
