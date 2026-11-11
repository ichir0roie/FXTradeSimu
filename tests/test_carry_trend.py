"""本命 2 本目(main_carry_trend、S039)と、共通の対策のスプレッドのゲート・mid の終値の損切り(guards)。"""
import numpy as np
import pytest

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.calendar import NS_DAY, server_hour
from fxsim.data import load_rates
from fxsim.data.market import MarketData
from fxsim.data.rates import MULTI_RATES
from fxsim.strategies import STRATEGIES
from fxsim.strategies.guards import GuardConfig, spread_ok, typical_spread


def test_typical_spread_is_causal_and_ignores_a_single_spike():
    s = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0])
    t = typical_spread(s, 3)
    assert t.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    s2 = s.copy()
    s2[5] = 100.0                                  # 後の足を変えても、前の足の典型は変わらない
    assert typical_spread(s2, 3)[:5].tolist() == t[:5].tolist()


def test_spread_ok():
    g = GuardConfig.from_dict({"max_spread_mult": 3.0})
    assert spread_ok(2.9, 1.0, g) and not spread_ok(3.1, 1.0, g)
    assert spread_ok(100.0, 1.0, GuardConfig())    # 0 なら見ない
    assert spread_ok(5.0, float("nan"), g)          # 典型が分からなければ止めない
    with pytest.raises(ValueError):
        GuardConfig.from_dict({"stop_price": "last"})


def test_server_hour_puts_the_rollover_at_midnight_in_summer_and_winter():
    summer = np.datetime64("2026-07-15T21:00", "ns").astype(np.int64)    # NY 17 時(夏時間)
    winter = np.datetime64("2026-12-15T22:00", "ns").astype(np.int64)    # NY 17 時(冬時間)
    assert server_hour(np.array([summer, winter]), "America/New_York+7").tolist() == [0, 0]


def _daily(close, pair="usdjpy"):
    n = len(close)
    t0 = np.datetime64("2019-01-07T22:00", "ns").astype(np.int64)
    times = t0 + np.arange(n, dtype=np.int64) * NS_DAY
    md = MarketData(times=times, mid=np.asarray(close, float), spread=np.full(n, 0.004), pip=0.01,
                    step_minutes=1440, meta={"swap_pair": pair})
    return md


def test_carry_trend_only_holds_the_receiving_side_and_needs_the_trend():
    close = np.r_[np.linspace(100, 110, 60), np.linspace(110, 95, 60)]
    md = _daily(close)
    b = md.bars(1440)
    diff = load_rates(MULTI_RATES).pair_differential("usdjpy", b.t)
    s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 0})
    target, stop, move = s.signals(b, np.random.default_rng(1))
    assert np.array_equal(target, np.sign(diff).astype(int))            # 金利差の符号の向きだけ
    s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 10})
    target, _, _ = s.signals(b, np.random.default_rng(1))
    side = int(np.sign(diff[0]))
    ma = np.convolve(b.close, np.ones(10) / 10, "full")[:len(b)]
    for i in range(10, len(b)):
        agree = np.sign(b.close[i] - ma[i]) == side
        assert target[i] == (side if agree else 0)
    s = STRATEGIES.create("main_carry_trend", {"min_carry": 1.0})      # 年 100% の金利差は無い
    assert not s.signals(b, np.random.default_rng(1))[0].any()


def test_spread_gate_blocks_new_entries_when_the_spread_is_wide():
    md = _daily(np.linspace(100, 130, 120))
    results = {}
    for mult in (0.0, 0.5):     # 0.5 倍: 今のスプレッドは必ず典型の 0.5 倍を超える
        s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 0,
                                                   "guard": {"max_spread_mult": mult, "cost_multiple": 0}})
        bt = build_backtester(md, s, BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}}),
                              EngineConfig(), seed=1)
        bt.run()
        results[mult] = (len(bt.deals), s.blocked["spread"])
    assert results[0.0][0] > 0 and results[0.0][1] == 0
    assert results[0.5][0] == 0 and results[0.5][1] > 0
