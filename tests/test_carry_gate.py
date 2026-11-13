"""main_carry_gate(S043): 関門を外せば main_carry_breakout と同じ、同時に持つ数を守る、関門は先を見ない。"""
import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from tests.test_carry_trend import _fake_pairs

BASE = {"n_in": 20, "n_out": 10, "min_carry": -1.0, "c2r_min": 0.0, "vol_n": 20, "protect": 0.0,
        "max_spread_mult": 0.0}


def _run(md, name, extra=None):
    s = STRATEGIES.create(name, {**BASE, **(extra or {})})
    bt = build_backtester(md, s, BrokerConfig.preset("oanda"),
                          EngineConfig(initial_balance=1e5, min_units=1000, swap_side_only=True), seed=1)
    bt.run()
    return s, bt


def test_without_gates_it_is_the_carry_breakout(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, a = _run(md, "main_carry_breakout")
    _, b = _run(md, "main_carry_gate")
    assert len(a.trades) > 0
    assert [(t.pair, t.entry_t, t.pnl) for t in a.trades] == [(t.pair, t.entry_t, t.pnl) for t in b.trades]


def test_max_held_caps_the_number_of_open_pairs(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, free = _run(md, "main_carry_gate")
    s, bt = _run(md, "main_carry_gate", {"max_held": 1})
    spans = [(t.entry_t, t.exit_t) for t in bt.trades]
    for a, _ in spans:
        assert sum(1 for x, y in spans if x <= a < y) <= 1
    both = [(t.entry_t, t.exit_t) for t in free.trades]
    if any(sum(1 for x, y in both if x <= a < y) > 1 for a, _ in both):
        assert s.blocked["held"] > 0


def test_gates_do_not_look_ahead(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    legs = md.meta["legs"]
    full_bars = {p: legs[p].bars(1440) for p in legs}
    params = {**BASE, "entry_vol_max": 1.1, "vol_window": 50, "counter_z": 0.3}
    full = STRATEGIES.create("main_carry_gate", params)
    full.prepare(full_bars, np.random.default_rng(0))
    n = len(next(iter(full_bars.values())).t)
    for stop in (n // 3, n // 2):
        part = STRATEGIES.create("main_carry_gate", params)
        part.prepare({p: b.slice(stop) for p, b in full_bars.items()}, np.random.default_rng(0))
        for p in full_bars:
            np.testing.assert_array_equal(part.vol_rel[p], full.vol_rel[p][:stop])
            np.testing.assert_array_equal(part.trend_z[p], full.trend_z[p][:stop])
