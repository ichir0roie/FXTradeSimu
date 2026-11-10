"""読み出し。集計・レポート・ケースの記録はここを通して DB を読む。"""
from __future__ import annotations

import json
import sqlite3


def get_run(con: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def latest_run_id(con: sqlite3.Connection) -> int | None:
    return con.execute("SELECT MAX(id) FROM runs").fetchone()[0]


def resolve_run(con: sqlite3.Connection, ref: str | int | None) -> int | None:
    """番号か run_key(前方一致も可)から、DB の番号を返す。"""
    if ref is None:
        return None
    s = str(ref)
    if s.isdigit():
        return int(s)
    row = con.execute("SELECT id FROM runs WHERE run_key = ?", (s,)).fetchone()
    if row is None:
        rows = con.execute("SELECT id FROM runs WHERE run_key LIKE ? ORDER BY id", (s + "%",)).fetchall()
        if len(rows) != 1:
            raise KeyError(f"run '{s}' が見つからないか、複数に当たります")
        row = rows[0]
    return int(row["id"])


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


def cost_summary(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """ブローカー x 戦略ごとの費用の内訳(円、パターン平均)。費用の内訳を記録していない古い run は空。"""
    q = """SELECT j.broker, j.strategy_label, COUNT(*) AS n,
                  AVG(json_extract(j.metrics_json, '$.costs.gross_pnl')) AS gross_pnl,
                  AVG(json_extract(j.metrics_json, '$.costs.spread')) AS spread,
                  AVG(json_extract(j.metrics_json, '$.costs.slippage')) AS slippage,
                  AVG(json_extract(j.metrics_json, '$.costs.commission')) AS commission,
                  AVG(json_extract(j.metrics_json, '$.costs.swap')) AS swap,
                  AVG(j.pnl) AS pnl,
                  AVG(json_extract(j.metrics_json, '$.hunted')) AS hunted,
                  AVG(json_extract(j.metrics_json, '$.losscut')) AS losscut,
                  AVG(json_extract(j.metrics_json, '$.rejected')) AS rejected,
                  AVG(json_extract(j.metrics_json, '$.zero_cut')) AS zero_cut
           FROM jobs j WHERE j.run_id = ? AND j.status = 'done'
             AND json_extract(j.metrics_json, '$.costs') IS NOT NULL
           GROUP BY j.broker, j.strategy_label ORDER BY j.broker, j.strategy_label"""
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


def done_jobs_with_metrics(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """完了したジョブの損益と成績の JSON、パターン(経路)の情報。検証の判定の材料。"""
    q = """SELECT j.id, j.strategy_label, j.pnl, j.ruined, j.metrics_json, p.id AS path_id, p.market, p.market_label,
                  p.seed, p.t_start, p.t_end
           FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ? AND j.status = 'done'"""
    return [dict(r) for r in con.execute(q, (run_id,))]


def case_test_ids(con: sqlite3.Connection) -> list[str]:
    """実行の記録があるテストケースの ID(fxsim verify が作ったものも含む)。"""
    return [r[0] for r in con.execute("SELECT DISTINCT test_id FROM case_runs ORDER BY test_id")]


# ---- 概要(docs/results)の材料 ------------------------------------------------

def summary_paths(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """相場パターン(経路)ごとの性質と現実らしさ。"""
    q = """SELECT id AS path_id, market_label, market, market_version, params_hash, seed, detail, status,
                  t_start, t_end, realism_score, n_fail, facts_json
           FROM paths WHERE run_id = ? ORDER BY market_label, seed, id"""
    return [dict(r) for r in con.execute(q, (run_id,))]


def summary_jobs(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """バックテストごとの成績(資産曲線・取引の明細は含まない)。"""
    q = """SELECT j.id AS job_id, j.path_id, p.market_label, p.seed, j.broker, j.strategy_label, j.strategy,
                  j.strategy_version, j.params_hash, j.timeframe, j.status, j.note, j.n_trades, j.pnl, j.metrics_json
           FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ?
           ORDER BY p.market_label, p.seed, j.broker, j.strategy_label, j.id"""
    return [dict(r) for r in con.execute(q, (run_id,))]


def case_runs_of(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """その実行に対応するテストケースの記録(定義のハッシュ・コードの版・要約)。"""
    q = """SELECT test_id, behavior_id, test_hash, behavior_hash, code_sha, status, created_at, finished_at,
                  summary_json
           FROM case_runs WHERE run_id = ? ORDER BY id"""
    return [dict(r) for r in con.execute(q, (run_id,))]
