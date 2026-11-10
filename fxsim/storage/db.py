"""SQLite への保存。シミュレーション中も逐次書き込み、Web UI(web/)がそれを読む。

構造:
    runs           実験 1 回(設定・git の版・状態)
    paths          生成した市場の経路 1 本(市場の案 x シード)。統計的性質(facts)と現実らしさの点数
    pattern_data   相場パターンの中身の目録。同じモデル・パラメータ・シード・期間なら実験をまたいで 1 つだけ持つ
    candles        相場パターンのローソク足(詳細保存するシードだけ。data_key で引く)
    market_events  相場パターンに入れた悪意のイベント(ストップ狩り・ダマシ・指標など。詳細保存のみ)
    jobs           経路 x ブローカー x 戦略 のバックテスト 1 本。進捗と最終指標
    equity         ジョブの資産曲線(詳細保存はローソク足と同じ間隔、ほかは日次)
    trades         ジョブの取引(既定では詳細保存のみ)
    case_defs      動作ケース・テストケースの定義のスナップショット
    case_runs      テストケースの実行記録(どの定義の版を、どの run で回し、要約はどうだったか)
集計のビュー:
    v_market_summary    実験 x 市場の案ごとに、シードを平均した facts と点数
    v_strategy_summary  実験 x 市場 x ブローカー x 戦略ごとに、シードを平均した成績
時刻はすべて UNIX 秒(UTC)。
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "fxsim.sqlite"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'experiment',
    proposal TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    git_sha TEXT,
    years REAL,
    seeds_json TEXT,
    config_json TEXT,
    notes TEXT,
    elapsed_sec REAL
);
CREATE TABLE IF NOT EXISTS paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    market TEXT NOT NULL,
    market_label TEXT NOT NULL,
    market_version TEXT,
    params_hash TEXT,
    params_json TEXT,
    seed INTEGER NOT NULL,
    detail INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    candle_tf INTEGER,
    t_start INTEGER,
    t_end INTEGER,
    facts_json TEXT,
    realism_score REAL,
    n_fail INTEGER,
    data_key TEXT                     -- 相場パターンの中身のキー(同じ中身なら実験をまたいで共有)
);
CREATE INDEX IF NOT EXISTS ix_paths_run ON paths(run_id);
-- 相場パターンの中身(ローソク足・イベント)。同じモデル・パラメータ・シード・期間なら 1 つだけ持つ
CREATE TABLE IF NOT EXISTS pattern_data (
    data_key TEXT PRIMARY KEY,
    market TEXT, seed INTEGER, candle_tf INTEGER,
    t_start INTEGER, t_end INTEGER,
    n_candles INTEGER, n_events INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS candles (
    data_key TEXT NOT NULL,
    t INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL,
    spread REAL,
    PRIMARY KEY (data_key, t)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS market_events (
    data_key TEXT NOT NULL,
    t INTEGER NOT NULL,
    kind TEXT NOT NULL,
    direction INTEGER,
    pips REAL,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_key_t ON market_events(data_key, t);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    path_id INTEGER NOT NULL REFERENCES paths(id) ON DELETE CASCADE,
    broker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_label TEXT NOT NULL,
    strategy_version TEXT,
    params_hash TEXT,
    params_json TEXT,
    schedule_json TEXT,
    timeframe INTEGER,
    status TEXT NOT NULL,
    bars_total INTEGER,
    bars_done INTEGER DEFAULT 0,
    sim_t INTEGER,
    balance REAL,
    equity REAL,
    n_trades INTEGER DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT,
    metrics_json TEXT,
    sharpe REAL, cagr REAL, max_dd REAL, profit_factor REAL, win_rate REAL, trades_per_year REAL,
    net_pips_per_year REAL, gross_pips_per_year REAL, cost_pips_per_year REAL, gross_sharpe REAL,
    final_equity REAL, pnl REAL, ruined INTEGER, hunted INTEGER, note TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_run ON jobs(run_id);
CREATE INDEX IF NOT EXISTS ix_jobs_path ON jobs(path_id);
CREATE TABLE IF NOT EXISTS equity (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    t INTEGER NOT NULL,
    balance REAL,
    equity REAL,
    PRIMARY KEY (job_id, t)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS trades (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    side INTEGER, units REAL,
    entry_t INTEGER, exit_t INTEGER,
    entry_price REAL, exit_price REAL,
    pnl REAL, pips REAL, gross_pips REAL, cost_pips REAL, swap REAL,
    reason TEXT, bars_held INTEGER,
    PRIMARY KEY (job_id, seq)
) WITHOUT ROWID;

-- 動作ケース・テストケースの定義(実行時点のスナップショット)と、テストケースの実行記録
CREATE TABLE IF NOT EXISTS case_defs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- behavior | test
    name TEXT,
    path TEXT,
    definition_json TEXT,
    hash TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS case_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT NOT NULL,
    behavior_id TEXT NOT NULL,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    test_hash TEXT,
    behavior_hash TEXT,
    code_sha TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    summary_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_case_runs_test ON case_runs(test_id);

DROP VIEW IF EXISTS v_market_summary;
CREATE VIEW v_market_summary AS
SELECT p.run_id, r.name AS run_name, p.market_label, p.market, p.market_version, p.params_hash,
       COUNT(*) AS n_seeds, AVG(p.realism_score) AS realism_score, AVG(p.n_fail) AS n_fail,
       AVG(json_extract(p.facts_json, '$.annual_vol')) AS annual_vol,
       AVG(json_extract(p.facts_json, '$.kurt_h1')) AS kurt_h1,
       AVG(json_extract(p.facts_json, '$.acf_abs_h1_lag1')) AS acf_abs_h1_lag1,
       AVG(json_extract(p.facts_json, '$.vr_h1_d1')) AS vr_h1_d1,
       AVG(json_extract(p.facts_json, '$.daily_range_pips')) AS daily_range_pips,
       AVG(json_extract(p.facts_json, '$.spread_p99_pips')) AS spread_p99_pips,
       AVG(json_extract(p.facts_json, '$.false_breakout_rate')) AS false_breakout_rate,
       AVG(json_extract(p.facts_json, '$.wick_rejection_rate')) AS wick_rejection_rate
FROM paths p JOIN runs r ON r.id = p.run_id
WHERE p.status = 'done'
GROUP BY p.run_id, p.market_label;

DROP VIEW IF EXISTS v_strategy_summary;
CREATE VIEW v_strategy_summary AS
SELECT j.run_id, r.name AS run_name, p.market_label, j.broker, j.strategy_label, j.strategy,
       j.strategy_version, j.params_hash, j.timeframe,
       COUNT(*) AS n_seeds,
       AVG(j.sharpe) AS sharpe, MIN(j.sharpe) AS sharpe_min, MAX(j.sharpe) AS sharpe_max,
       AVG(j.gross_sharpe) AS gross_sharpe,
       AVG(j.cagr) AS cagr, AVG(j.max_dd) AS max_dd, MAX(j.max_dd) AS max_dd_worst,
       AVG(j.profit_factor) AS profit_factor, AVG(j.win_rate) AS win_rate,
       AVG(j.trades_per_year) AS trades_per_year,
       AVG(j.net_pips_per_year) AS net_pips_per_year,
       AVG(j.gross_pips_per_year) AS gross_pips_per_year,
       AVG(j.cost_pips_per_year) AS cost_pips_per_year,
       AVG(j.final_equity) AS final_equity, AVG(j.pnl) AS pnl_mean, SUM(j.pnl) AS pnl_total,
       SUM(j.ruined) AS n_ruined, SUM(j.hunted) AS hunted
FROM jobs j JOIN paths p ON p.id = j.path_id JOIN runs r ON r.id = j.run_id
WHERE j.status = 'done'
GROUP BY j.run_id, p.market_label, j.broker, j.strategy_label;
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    con = connect(path)
    con.executescript(SCHEMA)
    return con


class Tx:
    """BEGIN IMMEDIATE ... COMMIT。書き込みが他プロセスと重なっても待って再試行する。"""

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def __enter__(self):
        for attempt in range(50):
            try:
                self.con.execute("BEGIN IMMEDIATE")
                return self.con
            except sqlite3.OperationalError as e:  # pragma: no cover - 競合時のみ
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                time.sleep(0.05 * (attempt + 1))
        raise sqlite3.OperationalError("could not acquire write lock")

    def __exit__(self, exc_type, exc, tb):
        self.con.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def dumps(obj) -> str:
    def default(o):
        try:
            import numpy as np

            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:  # pragma: no cover
            pass
        return str(o)

    return json.dumps(obj, ensure_ascii=False, default=default, allow_nan=True)


def checkpoint(con: sqlite3.Connection) -> None:
    """WAL をメインのファイルへ書き戻す(コミット前に呼ぶ)。"""
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
