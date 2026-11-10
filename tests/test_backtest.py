from dataclasses import dataclass

import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig, EngineConfig, compute_metrics
from fxsim.backtest.lookahead import check_lookahead
from fxsim.calendar import NS_HOUR
from fxsim.data import Bars
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES, Strategy

T0 = int(np.datetime64("2021-03-01T00:00", "ns").astype(np.int64))


def make(ohlc, spread=0.02):
    a = np.asarray(ohlc, dtype=float)
    o, h, l, c = a.T
    n = len(a)
    s = np.full(n, spread)
    return Bars(60, T0 + np.arange(n) * NS_HOUR, o, h, l, c, o - s / 2, h - s / 2, l - s / 2, c - s / 2,
                o + s / 2, h + s / 2, l + s / 2, c + s / 2, s, s, np.ones(n, dtype=np.int64), np.arange(n))


class Script(Strategy):
    """テスト用: 指定の足で指定の注文を出す。"""
    name = "script"

    @dataclass
    class Params:
        timeframe: int = 60

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def on_bar(self, i, ctx):
        for act in self.plan.get(i, []):
            getattr(ctx, act[0])(**act[1])


NOSLIP = BrokerConfig(slippage_pips=0.0)


def test_market_entry_fills_at_next_open_ask_and_sl_hits():
    bars = make([[100, 100.1, 99.9, 100], [100.0, 100.05, 99.5, 99.6], [99.6, 99.7, 99.5, 99.6]])
    bt = Backtester(bars, Script({0: [("buy", {"sl": 0.2})]}), NOSLIP, pip=0.01).run()
    t = bt.trades[0]
    assert t.entry_price == pytest.approx(100.01)       # ask の始値
    assert t.reason == "sl"
    assert t.exit_price == pytest.approx(99.81)         # 損切り価格(bid 基準)
    assert t.pips == pytest.approx(-20.0)


def test_gap_through_stop_fills_at_open():
    bars = make([[100, 100, 100, 100], [100, 100, 100, 100], [99.0, 99.1, 98.9, 99.0]])
    bt = Backtester(bars, Script({0: [("buy", {"sl": 0.2})]}), NOSLIP, pip=0.01).run()
    assert bt.trades[0].exit_price == pytest.approx(99.0 - 0.01)


def test_sl_before_tp_in_same_bar():
    bars = make([[100, 100, 100, 100], [100, 101, 99, 100]])
    bt = Backtester(bars, Script({0: [("sell", {"sl": 0.3, "tp": 0.3})]}), NOSLIP, pip=0.01).run()
    assert bt.trades[0].reason == "sl"


def test_tp_gap_kept_by_malicious_broker():
    bars = make([[100, 100, 100, 100], [100, 100, 100, 100], [101.0, 101, 101, 101]])
    plan = {0: [("buy", {"tp": 0.3})]}
    fair = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0), pip=0.01).run().trades[0]
    evil_broker = BrokerConfig(slippage_pips=0, positive_slippage=False)
    evil = Backtester(bars, Script(plan), evil_broker, pip=0.01).run().trades[0]
    assert fair.exit_price > evil.exit_price
    assert evil.exit_price == pytest.approx(100.01 + 0.3)


def test_hunting_broker_stops_out_near_misses():
    bars = make([[100, 100, 100, 100], [100, 100.05, 99.84, 100]] + [[100, 100.05, 99.84, 100]] * 20)
    plan = {0: [("buy", {"sl": 0.2})]}
    evil = BrokerConfig(slippage_pips=0, hunt_prob=1.0, hunt_pips=5)
    bt = Backtester(bars, Script(plan), evil, pip=0.01).run()
    assert bt.trades[0].reason == "hunted"
    safe = Backtester(bars, Script(plan), NOSLIP, pip=0.01).run()
    assert safe.trades[0].reason == "end"


def test_swap_is_charged_with_triple_wednesday():
    days = 8
    n = days * 24
    bars = make([[100, 100, 100, 100]] * n)
    br = BrokerConfig(slippage_pips=0, swap_long_per_10k=10.0)
    bt = Backtester(bars, Script({0: [("buy", {})]}), br, EngineConfig(units=10_000), pip=0.01).run()
    # 2021-03-01 は月曜。22:00 UTC のロールオーバーが 8 回: 月火水(3)木金 = 7、土日 = 0、翌月曜 = 1
    assert bt.trades[0].swap == pytest.approx(10.0 * 8)


def test_risk_sizing_and_leverage_cap():
    bars = make([[100, 100, 100, 100]] * 5)
    cfg = EngineConfig(sizing="risk", risk_fraction=0.01, initial_balance=1_000_000)
    bt = Backtester(bars, Script({0: [("buy", {"sl": 0.5})]}), NOSLIP, cfg, pip=0.01).run()
    assert bt.trades[0].units == 20_000            # 1 万円 / 0.5 円
    cfg = EngineConfig(units=10_000_000, max_units=10_000_000)
    bt = Backtester(bars, Script({0: [("buy", {})]}), NOSLIP, cfg, pip=0.01).run()
    assert bt.trades[0].units <= 1_000_000 * 25 / 100.01


@pytest.mark.parametrize("name", ["random_entry", "ma_cross", "donchian_breakout", "bollinger_reversion",
                                  "simple_follower", "buy_and_hold"])
def test_builtin_strategies_do_not_look_ahead(name):
    md = MARKETS.create("malicious").generate(days=120, seed=4)
    tf = STRATEGIES.create(name).timeframe
    assert check_lookahead(lambda: STRATEGIES.create(name), md.bars(tf)) == []


def test_metrics_cost_decomposition():
    md = MARKETS.create("stylized").generate(days=200, seed=1)
    bt = Backtester(md.bars(60), STRATEGIES.create("random_entry"), BrokerConfig.preset("malicious"),
                    pip=md.pip).run()
    m = compute_metrics(bt)
    assert m["n_trades"] > 10
    assert m["cost_pips_per_year"] > 0
    assert m["gross_pips_per_year"] - m["cost_pips_per_year"] == pytest.approx(m["net_pips_per_year"], rel=1e-6)
