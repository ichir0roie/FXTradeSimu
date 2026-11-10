import sqlite3
from dataclasses import dataclass

import numpy as np

from fxsim.backtest import Backtester, BrokerConfig
from fxsim.backtest.schedule import TradingSchedule
from fxsim.calendar import NS_HOUR
from fxsim.cases import load_cases, run_cases, to_experiment
from fxsim.markets import MARKETS
from fxsim.storage import init_db
from fxsim.strategies import Strategy


class AlwaysLong(Strategy):
    name = "always_long"

    @dataclass
    class Params:
        timeframe: int = 60

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_bar(self, i, ctx):
        self.calls.append(i)
        if ctx.side == 0:
            ctx.buy()


def test_schedule_masks_hours_days_and_interval():
    t0 = int(np.datetime64("2021-03-01T00:00", "ns").astype(np.int64))   # 月曜
    t = t0 + np.arange(24 * 7) * NS_HOUR
    s = TradingSchedule(decision_every=2, hours=[["09:00", "15:00"]], tz="JST", outside="close")
    decide, can_enter, force_close = s.masks(t, 60)
    close_jst = ((t // NS_HOUR) + 1 + 9) % 24
    assert np.all(can_enter[(close_jst >= 9) & (close_jst < 15) & (np.arange(len(t)) < 24 * 4)])
    assert not np.any(can_enter[(close_jst < 9) | (close_jst >= 15)])
    assert np.array_equal(force_close, ~can_enter)
    assert decide.sum() == len(t) // 2


def test_engine_respects_schedule():
    md = MARKETS.create("random_walk").generate(days=20, seed=1)
    bars = md.bars(60)
    sched = TradingSchedule(decision_every=4, hours=[["08:00", "12:00"]], outside="close")
    strat = AlwaysLong()
    bt = Backtester(bars, strat, BrokerConfig(slippage_pips=0), pip=md.pip, schedule=sched).run()
    assert all(i % 4 == 0 for i in strat.calls)
    hours = [(t.entry_t // NS_HOUR) % 24 for t in bt.trades]
    assert set(hours) <= {8, 9, 10, 11}
    assert {t.reason for t in bt.trades} <= {"signal", "end"}


def test_repository_cases_are_valid_and_map_to_experiments():
    behaviors, tests = load_cases()
    assert tests and behaviors
    for t in tests.values():
        cfg = to_experiment(t, behaviors[t.behavior])
        assert cfg.kind == "test_case" and cfg.proposal == t.id
        assert [m.seeds for m in cfg.markets] == [p.seeds for p in t.patterns]


def test_run_cases_records_and_skips_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    (root / "behaviors").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "behaviors" / "B1.yaml").write_text(
        "id: B9\nstrategy: ma_cross\nparams: {fast: 5, slow: 20}\ntimeframe: 1440\nbroker: fair\n", encoding="utf-8")
    (root / "tests" / "T1.yaml").write_text(
        "id: T9\nbehavior: B9\nperiod: {start: '2024-01-01', end: '2024-06-30'}\n"
        "patterns:\n  - {market: real}\n  - {market: real_bridge, seeds: '1-2'}\n", encoding="utf-8")
    import fxsim.cases as cases

    monkeypatch.setattr(cases, "CASES_DIR", root)
    db = tmp_path / "t.sqlite"
    con = init_db(db)
    log = []
    ids = run_cases(con, str(db), ["T9"], log=log.append)
    assert len(ids) == 1
    row = con.execute("SELECT status, summary_json FROM case_runs WHERE test_id='T9'").fetchone()
    assert row[0] == "done" and '"by_market"' in row[1]
    n_paths = con.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
    assert n_paths == 3   # real 1 本 + real_bridge 2 本
    t0 = con.execute("SELECT MIN(t_start) FROM paths WHERE market='real'").fetchone()[0]
    assert t0 >= int(np.datetime64("2024-01-01", "s").astype(np.int64))
    assert run_cases(con, str(db), ["T9"], log=log.append) == []   # 変わっていないので飛ばす
    assert any("飛ばします" in m for m in log)
    assert isinstance(con, sqlite3.Connection)
