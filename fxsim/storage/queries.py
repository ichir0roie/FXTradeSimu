"""読み出し。集計・レポート・ケースの記録はここを通して DB を読む。"""
from __future__ import annotations

import json
import sqlite3


def get_run(con: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def latest_run_id(con: sqlite3.Connection) -> int | None:
    return con.execute("SELECT MAX(id) FROM runs").fetchone()[0]


def run_status(con: sqlite3.Connection, run_id: int) -> str:
    return con.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0]


def market_summary(con: sqlite3.Connection, run_id: int | None = None, limit: int | None = None) -> list[sqlite3.Row]:
    """v_market_summary。run_id を省くと全実験(新しい順)。"""
    if run_id is not None:
        return con.execute("SELECT * FROM v_market_summary WHERE run_id=? ORDER BY market_label",
                           (run_id,)).fetchall()
    return con.execute("SELECT * FROM v_market_summary ORDER BY run_id DESC, market_label LIMIT ?",
                       (limit or -1,)).fetchall()


def strategy_ranking(con: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM v_strategy_summary ORDER BY sharpe DESC LIMIT ?", (limit,)).fetchall()


def job_rows(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """実験のジョブを、市場・シードの順に。パターンごとの統計の材料。"""
    q = """SELECT j.id, p.market_label, p.market, p.seed, j.broker, j.strategy_label, j.status, j.note,
                  j.pnl, j.net_pips_per_year, j.sharpe, j.max_dd, j.gross_sharpe, j.trades_per_year, j.win_rate,
                  j.cost_pips_per_year, j.gross_pips_per_year, j.final_equity, j.ruined
           FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ? ORDER BY p.market_label, p.seed"""
    return [dict(r) for r in con.execute(q, (run_id,))]


def skipped_jobs(con: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return con.execute("SELECT strategy_label, note, COUNT(*) n FROM jobs WHERE run_id=? AND status='skipped'"
                       " GROUP BY strategy_label, note", (run_id,)).fetchall()


def find_case_run(con: sqlite3.Connection, test_id: str, test_hash: str, behavior_hash: str,
                  code_sha: str | None) -> int | None:
    """同じ定義・同じコードで完了した実行があれば、その run 番号。"""
    row = con.execute("SELECT run_id FROM case_runs WHERE test_id=? AND test_hash=? AND behavior_hash=?"
                      " AND code_sha IS ? AND status='done' ORDER BY id DESC LIMIT 1",
                      (test_id, test_hash, behavior_hash, code_sha)).fetchone()
    return row[0] if row else None


def latest_case_run(con: sqlite3.Connection, test_id: str) -> tuple[int, dict] | None:
    """テストケースの最新の完了した実行(run 番号, 要約)。"""
    row = con.execute("SELECT run_id, summary_json FROM case_runs WHERE test_id=? AND status='done'"
                      " ORDER BY id DESC LIMIT 1", (test_id,)).fetchone()
    if not row:
        return None
    return row[0], json.loads(row[1]) if row[1] else {}
