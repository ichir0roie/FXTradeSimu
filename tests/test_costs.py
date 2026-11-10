"""費用と約定の現実らしさ: 経路での約定、手数料、金利差のスワップ、ラストルック、ゼロカット。"""
from dataclasses import dataclass

import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig, EngineConfig, compute_metrics
from fxsim.calendar import trading_times
from fxsim.data import load_rates, make_bars
from fxsim.strategies import Strategy

PIP = 0.01


class Script(Strategy):
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


def bars_from_minutes(prices, spread_pips=0.2, start="2021-03-01", spreads=None):
    """1 分ごとの mid から 1 時間足を作る(足の中の経路つき)。"""
    t = trading_times(start, days=len(prices) / 1440 + 3)[:len(prices)]
    mid = np.asarray(prices, dtype=float)
    spread = np.full(len(mid), spread_pips * PIP) if spreads is None else np.asarray(spreads) * PIP
    return make_bars(t, mid, spread, 60)


def flat_then(prices_after, start_price=100.0, n_before=120):
    return [start_price] * n_before + list(prices_after)


def test_stop_fills_at_the_price_after_a_crash_not_at_the_stop():
    # 2 本目の足の途中で 100.00 -> 98.00 に一気に落ちる(フラッシュクラッシュ)
    prices = flat_then([100.0] * 30 + [98.0] * 30 + [98.5] * 60, n_before=60)
    bars = bars_from_minutes(prices)
    broker = BrokerConfig(slippage_pips=0)
    bt = Backtester(bars, Script({0: [("buy", {"sl": 0.20})]}), broker, pip=PIP).run()
    t = bt.trades[0]
    assert t.reason == "sl"
    assert t.exit_price == pytest.approx(98.0 - 0.001)          # 損切り 99.80 ではなく、落ちた先の bid
    pess = BrokerConfig(slippage_pips=0, intrabar="pessimistic")
    t2 = Backtester(bars, Script({0: [("buy", {"sl": 0.20})]}), pess, pip=PIP).run().trades[0]
    assert t2.exit_price == pytest.approx(100.001 - 0.20)       # 足の高安だけだと損切り価格で約定(甘い)


def test_first_touch_wins_when_stop_and_target_are_in_the_same_bar():
    # 2 本目の足で先に +0.5 円上がり、その後 -0.5 円下がる -> 利確が先
    prices = flat_then([100.0] * 10 + [100.6] * 20 + [99.4] * 30, n_before=60)
    bars = bars_from_minutes(prices)
    plan = {0: [("buy", {"sl": 0.3, "tp": 0.3})]}
    path = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0), pip=PIP).run().trades[0]
    pess_broker = BrokerConfig(slippage_pips=0, intrabar="pessimistic")
    pess = Backtester(bars, Script(plan), pess_broker, pip=PIP).run().trades[0]
    assert path.reason == "tp" and pess.reason == "sl"


def test_commission_is_charged_on_both_sides():
    bars = bars_from_minutes([100.0] * 600)
    broker = BrokerConfig(slippage_pips=0, commission_per_10k=50.0)
    bt = Backtester(bars, Script({0: [("buy", {})], 3: [("close", {})]}), broker,
                    EngineConfig(units=20_000), pip=PIP).run()
    t = bt.trades[0]
    assert t.commission == pytest.approx(2 * 50.0 * 2)
    assert bt.balance == pytest.approx(1_000_000 + t.pnl)
    assert compute_metrics(bt)["costs"]["commission"] == pytest.approx(200.0)


def test_rate_swap_follows_the_rate_differential():
    bars = bars_from_minutes([150.0] * (1440 * 6), start="2023-08-07")   # 2023 年は米 5.38% / 日 -0.10%
    broker = BrokerConfig(slippage_pips=0, swap_mode="rates", swap_markup=0.0)
    long_ = Backtester(bars, Script({0: [("buy", {})]}), broker, EngineConfig(units=10_000), pip=PIP).run()
    short = Backtester(bars, Script({0: [("sell", {})]}), broker, EngineConfig(units=10_000), pip=PIP).run()
    diff = float(load_rates().differential(np.array([np.datetime64("2023-08-07", "ns").astype(np.int64)]))[0])
    per_day = 10_000 * 150.0 * diff / 365
    assert long_.trades[0].swap > 0 > short.trades[0].swap
    assert long_.trades[0].swap / per_day == pytest.approx(round(long_.trades[0].swap / per_day), abs=0.01)
    marked = Backtester(bars, Script({0: [("buy", {})]}), BrokerConfig(slippage_pips=0, swap_mode="rates",
                        swap_markup=0.01), EngineConfig(units=10_000), pip=PIP).run()
    assert marked.trades[0].swap < long_.trades[0].swap               # 上乗せの分だけ少ない


def test_spread_markup_and_floor_widen_fills():
    bars = bars_from_minutes([100.0] * 600, spread_pips=0.1)
    plan = {0: [("buy", {})], 3: [("close", {})]}
    base = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0), pip=PIP).run().trades[0]
    wide = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0, spread_markup_pips=0.3,
                                                       spread_floor_pips=1.0), pip=PIP).run().trades[0]
    assert base.cost_pips == pytest.approx(0.1)
    assert wide.cost_pips == pytest.approx(1.0)                        # 0.1 + 0.3 は下限 1.0 に届かない


def test_last_look_delays_favorable_entries():
    # 判断した足の終値より、次の足の始値が安い(買いに有利)
    prices = [100.0] * 60 + [99.9] * 600
    bars = bars_from_minutes(prices)
    broker = BrokerConfig(slippage_pips=0, last_look_prob=1.0)
    bt = Backtester(bars, Script({0: [("buy", {})], 4: [("close", {})]}), broker, pip=PIP).run()
    assert bt.stats["rejected"] >= 1
    assert bt.trades[0].entry_t > bars.t[1]


def test_zero_cut_absorbs_negative_balance_only_when_enabled():
    prices = [100.0] * 120 + [60.0] * 600                               # 40% の暴落で証拠金を超えて負ける
    bars = bars_from_minutes(prices)
    plan = {0: [("buy", {"units": 200_000})]}
    cfg = EngineConfig(max_units=1_000_000)
    owed = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0, zero_cut=False), cfg, pip=PIP).run()
    cut = Backtester(bars, Script(plan), BrokerConfig(slippage_pips=0, zero_cut=True), cfg, pip=PIP).run()
    assert owed.balance < 0 and owed.ruined
    assert cut.balance == 0 and cut.stats["zero_cut"] > 0 and cut.ruined
    assert owed.trades[0].reason == "losscut"
