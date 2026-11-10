import sqlite3

from fxsim.experiments import ExperimentConfig, run_experiment
from fxsim.experiments.report import run_report
from fxsim.experiments.stats import strategy_stats, summarize


def test_summarize():
    s = summarize([1.0, -2.0, 3.0, None])
    assert s["n"] == 3 and s["total"] == 2.0 and s["median"] == 1.0
    assert abs(s["positive_ratio"] - 2 / 3) < 1e-12


def test_end_to_end_writes_sqlite(tmp_path):
    db = tmp_path / "t.sqlite"
    cfg = ExperimentConfig.from_dict({
        "name": "t", "years": 0.25, "seeds": [1, 2], "workers": 1,
        "markets": ["random_walk", "real", "real_bridge_malicious"],
        "brokers": ["fair", "malicious"],
        "strategies": ["random_entry", {"name": "ma_cross", "label": "ma_d1", "params": {"timeframe": 1440,
                                                                                          "fast": 5, "slow": 20}}],
    })
    run_id = run_experiment(cfg, db, log=lambda *a: None)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    assert con.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0] == "done"
    # real は決定的なのでシード 1 本だけ
    assert con.execute("SELECT COUNT(*) FROM paths WHERE market='real'").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM paths").fetchone()[0] == 5
    # 日足しか無い real では 1 時間足の戦略はスキップ
    assert con.execute("SELECT COUNT(*) FROM jobs j JOIN paths p ON p.id=j.path_id WHERE p.market='real' AND"
                       " j.status='skipped'").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM jobs WHERE status NOT IN ('done','skipped')").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM candles").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM equity").fetchone()[0] > 0
    stats = strategy_stats(con, run_id)
    assert any(s["market"] == "ALL" for s in stats)
    assert "戦略の損益" in run_report(con, run_id)


def test_identical_patterns_share_stored_data(tmp_path):
    db = tmp_path / "s.sqlite"
    d = {"name": "a", "years": 0.1, "seeds": [1], "workers": 1, "markets": ["malicious"],
         "strategies": ["random_entry"]}
    run_experiment(ExperimentConfig.from_dict(d), db, log=lambda *a: None)
    run_experiment(ExperimentConfig.from_dict({**d, "name": "b", "strategies": ["ma_cross"]}), db, log=lambda *a: None)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM paths").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(DISTINCT data_key) FROM paths").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM pattern_data").fetchone()[0] == 1
