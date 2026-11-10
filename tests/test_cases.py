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
    book = load_cases()
    assert book.tests and book.behaviors
    for t in book.tests.values():
        cfg = to_experiment(t, book.behaviors[t.behavior])
        assert cfg.kind == "test_case" and cfg.proposal == t.id
        assert [m.seeds for m in cfg.markets] == [p.seeds for p in t.patterns]


def test_run_cases_records_and_skips_unchanged(tmp_path):
    root = tmp_path / "cases"
    (root / "behaviors").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "behaviors" / "B1.yaml").write_text(
        "id: B9\nstrategy: ma_cross\nparams: {fast: 5, slow: 20}\ntimeframe: 1440\nbroker: fair\n", encoding="utf-8")
    (root / "tests" / "T1.yaml").write_text(
        "id: T9\nbehavior: B9\nperiod: {start: '2024-01-01', end: '2024-06-30'}\n"
        "patterns:\n  - {market: real}\n  - {market: real_bridge, seeds: '1-2'}\n", encoding="utf-8")
    db = tmp_path / "t.sqlite"
    con = init_db(db)
    log = []
    ids = run_cases(con, str(db), ["T9"], root=root, log=log.append)
    assert len(ids) == 1
    row = con.execute("SELECT status, summary_json FROM case_runs WHERE test_id='T9'").fetchone()
    assert row[0] == "done" and '"by_market"' in row[1]
    n_paths = con.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
    assert n_paths == 3   # real 1 本 + real_bridge 2 本
    t0 = con.execute("SELECT MIN(t_start) FROM paths WHERE market='real'").fetchone()[0]
    assert t0 >= int(np.datetime64("2024-01-01", "s").astype(np.int64))
    assert run_cases(con, str(db), ["T9"], root=root, log=log.append) == []   # 変わっていないので飛ばす
    assert any("飛ばします" in m for m in log)
    assert isinstance(con, sqlite3.Connection)


def test_verification_runs_paired_null_and_records_verdict(tmp_path):
    from fxsim.cases import run_cases, verification_case
    from fxsim.storage import queries

    db = tmp_path / "v.sqlite"
    con = init_db(db)
    t = verification_case("B004", years=1.0, seeds=[1, 2])
    t.workers = 1
    assert t.verify and {p.market for p in t.patterns} >= {"real", "real_bridge", "real_bridge_placebo"}
    run_cases(con, str(db), [t.id], extra=[t], log=lambda *a: None)
    run_id, summary = queries.latest_case_run(con, t.id)
    v = summary["verdict"]
    assert v["verdict"] in ("合格", "条件付き", "不合格")
    assert [c["key"] for c in v["checks"]] == ["C1", "C2", "C3", "C4"]
    assert {"real", "bootstrap", "placebo"} <= set(v["groups"])
    # 帰無(NULL)は対になる同じパターンで回っている
    rows = con.execute("SELECT strategy_label, COUNT(*) FROM jobs WHERE run_id=? GROUP BY 1", (run_id,)).fetchall()
    counts = dict(rows)
    assert counts["NULL"] == counts["B004"]
    # 要約の統計は帰無を含まない
    not_done = con.execute("SELECT COUNT(*) FROM jobs WHERE run_id=? AND strategy_label='B004' AND status!='done'",
                           (run_id,)).fetchone()[0]
    assert summary["all"]["pnl"]["n"] == counts["B004"] - not_done


def test_direction_null_keeps_timing_and_randomizes_side():
    from fxsim.backtest.lookahead import check_lookahead
    from fxsim.strategies import STRATEGIES

    md = MARKETS.create("random_walk").generate(days=200, seed=3)
    bars = md.bars(60)
    bt = Backtester(bars, STRATEGIES.create("direction_null", {"inner": "donchian_breakout"}),
                    BrokerConfig(slippage_pips=0), pip=md.pip, seed=1).run()
    sides = [t.side for t in bt.trades]
    assert len(sides) > 20 and 0.2 < sides.count(1) / len(sides) < 0.8
    assert check_lookahead(lambda: STRATEGIES.create("direction_null", {"inner": "ma_cross"}), bars) == []


def test_verification_passes_a_model_with_a_real_edge(tmp_path):
    """陽性対照: 「実データ」に既知の優位性(日次リターンの自己相関 0.3)を仕込み、それに乗る戦略なら合格になる。

    判定がどんなモデルも落とす仕組みではないこと、そして偽薬(符号でたらめ)ではその優位性が消えることの確認。
    """
    import numpy as np

    from fxsim.cases import run_cases, verification_case
    from fxsim.storage import queries

    rng = np.random.default_rng(0)
    days = np.busday_offset("2015-01-05", np.arange(2400), roll="forward")
    r = np.zeros(len(days))
    for k in range(1, len(r)):
        r[k] = 0.3 * r[k - 1] + 0.006 * rng.standard_normal()
    csv = tmp_path / "ar1.csv"
    csv.write_text("time,close\n" + "".join(f"{d},{100 * np.exp(x):.6f}\n" for d, x in zip(days, np.cumsum(r),
                                                                                         strict=True)))
    root = tmp_path / "cases"
    (root / "behaviors").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "behaviors" / "B.yaml").write_text(
        "id: BX\nstrategy: tsmom\nparams: {lookbacks: [1]}\ntimeframe: 1440\nbroker: malicious\n", encoding="utf-8")
    db = tmp_path / "o.sqlite"
    con = init_db(db)
    t = verification_case("BX", years=6.0, seeds=[1, 2, 3, 4, 5, 6], dataset=str(csv), block_days=60)
    t.workers = 2
    run_cases(con, str(db), [t.id], extra=[t], root=root, log=lambda *a: None)
    v = queries.latest_case_run(con, t.id)[1]["verdict"]
    assert v["verdict"] == "合格", v["checks"]
    assert v["groups"]["placebo"]["t"] < 2      # 偽薬では優位性が消えている
