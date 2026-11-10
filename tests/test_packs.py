"""結果パック(1 実行 = 1 ファイル)の書き出しと取り込み。"""
import sqlite3

import pytest

from fxsim.experiments import ExperimentConfig, run_experiment
from fxsim.storage import init_db, packs

CFG = {"name": "p", "proposal": "E999", "years": 0.1, "seeds": [1, 2], "workers": 1,
       "markets": ["random_walk", "malicious"], "brokers": ["fair"],
       "strategies": ["random_entry", "ma_cross"], "storage": {"trades": "all"}}


def _summary(db, run_id):
    con = sqlite3.connect(db)
    q = """SELECT p.market_label, p.seed, j.broker, j.strategy_label, j.pnl, j.n_trades,
                  (SELECT COUNT(*) FROM equity e WHERE e.job_id = j.id),
                  (SELECT COUNT(*) FROM trades t WHERE t.job_id = j.id)
           FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ? ORDER BY 1, 2, 3, 4"""
    return con.execute(q, (run_id,)).fetchall()


def test_export_and_import_round_trip(tmp_path):
    src = tmp_path / "src.sqlite"
    res, pat = tmp_path / "results", tmp_path / "patterns"
    run_id = run_experiment(ExperimentConfig.from_dict(CFG), src, log=lambda *a: None)
    con = init_db(src)
    key = con.execute("SELECT run_key FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert key.startswith("E999-")
    out = packs.export_run(con, run_id, res, pat)
    assert out.name == f"{key}.sqlite"
    assert len(list(pat.glob("*.sqlite"))) == con.execute("SELECT COUNT(*) FROM pattern_data").fetchone()[0] > 0

    dst = tmp_path / "dst.sqlite"
    con2 = init_db(dst)
    con2.execute("INSERT INTO runs(name, kind, status, created_at, run_key) VALUES ('x', 'experiment', 'done',"
                 " '2000-01-01T00:00:00Z', 'other')")      # 番号がずれても対応が取れること
    new = packs.sync_packs(con2, res, pat)
    assert len(new) == 1 and new[0] != run_id
    assert _summary(src, run_id) == _summary(dst, new[0])
    n_candles = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    assert con2.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == n_candles
    assert packs.sync_packs(con2, res, pat) == []                       # 2 回目は何もしない
    assert packs.resolve_run(con2, key[:12]) == new[0]                  # 前方一致でも引ける


def test_legacy_db_gets_stable_run_keys(tmp_path):
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
    a = packs.make_run_key("V-B007", "test_case", "2026-11-10T05:01:42Z")
    b = packs.make_run_key("V-B007", "test_case", "2026-11-10T05:01:42Z")
    assert a != b and a.startswith("V-B007-20261110T050142-")
    assert packs.make_run_key(None, "eval market/x", "2026-11-10T05:01:42Z").startswith("eval_market_x-")
    with pytest.raises(KeyError):
        packs.resolve_run(init_db(":memory:", sync=False), "nothing")
