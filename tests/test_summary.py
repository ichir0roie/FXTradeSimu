"""実験の概要(docs/results/<run_key>/)の書き出しと読み直し、run_key。"""
import sqlite3

import pytest

from fxsim.experiments import ExperimentConfig, run_experiment
from fxsim.experiments.stats import strategy_stats
from fxsim.reports import export_summary, read_summary
from fxsim.storage import init_db, queries
from fxsim.storage.db import make_run_key

CFG = {"name": "p", "proposal": "E999", "years": 0.1, "seeds": [1, 2], "workers": 1,
       "markets": ["random_walk", "malicious"], "brokers": ["fair"],
       "strategies": ["random_entry", "ma_cross"], "storage": {"trades": "all"}}


def test_summary_round_trip(tmp_path):
    db = tmp_path / "x.sqlite"
    run_id = run_experiment(ExperimentConfig.from_dict(CFG), db, log=lambda *a: None)
    con = init_db(db)
    key = con.execute("SELECT run_key FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert key.startswith("E999-")
    out = export_summary(con, run_id, tmp_path / "results")
    assert out.name == key and {p.name for p in out.iterdir()} == {"run.json", "paths.csv", "jobs.csv"}

    s = read_summary(key[:12], tmp_path / "results")          # 前方一致でも引ける
    assert s["run"]["proposal"] == "E999" and s["run"]["config"]["years"] == 0.1
    assert len(s["paths"]) == 4 and len(s["jobs"]) == 8
    by_job = {j["job_id"]: j for j in s["jobs"]}
    for r in con.execute("SELECT id, pnl, sharpe, n_trades, params_hash, metrics_json FROM jobs WHERE run_id=?",
                         (run_id,)):
        j = by_job[r["id"]]
        assert j["n_trades"] == r["n_trades"] and j["params_hash"] == r["params_hash"]
        assert j["pnl"] == pytest.approx(r["pnl"], abs=1e-3)
        assert j["sharpe"] == pytest.approx(r["sharpe"], rel=1e-6)
        assert "cost_spread" in j and isinstance(j["yearly_pnl"], dict)
    # 概要からパターン単位の統計を取り直せる(DB の統計と同じ損益の並び)
    pnl = sorted(round(j["pnl"], 2) for j in s["jobs"])
    assert pnl == sorted(round(r[0], 2) for r in con.execute("SELECT pnl FROM jobs WHERE run_id=?", (run_id,)))
    assert strategy_stats(con, run_id)                         # DB 側も壊れていない
    assert all(p["realism_score"] is not None for p in s["paths"])


def test_old_db_gets_stable_run_keys(tmp_path):
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'experiment', proposal TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL,
        finished_at TEXT, git_sha TEXT, years REAL, seeds_json TEXT, config_json TEXT, notes TEXT, elapsed_sec REAL);
        INSERT INTO runs(name, proposal, status, created_at) VALUES ('a', 'E005', 'done', '2026-11-10T04:56:32Z');""")
    con.close()
    con = init_db(db)
    assert con.execute("SELECT run_key FROM runs").fetchone()[0] == "E005-20261110T045632-r01"


def test_run_keys_are_unique_and_safe_for_file_names():
    a = make_run_key("V-B007", "test_case", "2026-11-10T05:01:42Z")
    b = make_run_key("V-B007", "test_case", "2026-11-10T05:01:42Z")
    assert a != b and a.startswith("V-B007-20261110T050142-")
    assert make_run_key(None, "eval market/x", "2026-11-10T05:01:42Z").startswith("eval_market_x-")
    with pytest.raises(KeyError):
        queries.resolve_run(init_db(":memory:"), "nothing")
