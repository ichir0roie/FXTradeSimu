"""entry_filter(S042): 条件を外せば中の戦略と同じ、条件は先を見ない、止めた足では新規が出ない。"""
import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES

PIP = 0.01


@pytest.fixture(scope="module")
def bars():
    return MARKETS.create("malicious").generate(years=1, seed=5).bars(60)


def _run(bars, strat):
    return Backtester(bars, strat, BrokerConfig.preset("fair"), pip=PIP, seed=1).run()


def test_without_conditions_it_is_the_inner_strategy(bars):
    a = _run(bars, STRATEGIES.create("ma_cross"))
    b = _run(bars, STRATEGIES.create("entry_filter", {"inner": "ma_cross"}))
    assert [t.pnl for t in a.trades] == [t.pnl for t in b.trades]


def test_conditions_do_not_look_ahead(bars):
    params = {"inner": "ma_cross", "max_vol_ratio": 1.1, "max_er": 0.3, "vol_window": 200}
    full = STRATEGIES.create("entry_filter", params)
    full.prepare(bars, np.random.default_rng(0))
    assert 0 < full.blocked.mean() < 1
    for stop in (300, 1000, len(bars.close) // 2):
        part = STRATEGIES.create("entry_filter", params)
        part.prepare(bars.slice(stop), np.random.default_rng(0))
        assert np.array_equal(part.blocked, full.blocked[:stop])


def test_no_entry_is_filled_after_a_blocked_bar(bars):
    s = STRATEGIES.create("entry_filter", {"inner": "ma_cross", "max_vol_ratio": 1.1, "vol_window": 200})
    bt = _run(bars, s)
    assert s.n_blocked > 0
    for t in bt.trades:
        k = int(np.searchsorted(bars.t, t.entry_t, "right")) - 2      # 入る判断をした足
        assert not s.blocked[k]
