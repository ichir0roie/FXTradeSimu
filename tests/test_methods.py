"""よくある投資手法(S007〜S031)と、賭け方・ナンピンの仕組み。"""
from dataclasses import dataclass

import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig
from fxsim.calendar import trading_times
from fxsim.data import make_bars
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES, Strategy
from fxsim.strategies.builtin.gotobi import is_gotobi
from fxsim.strategies.staking import Staker

PIP = 0.01
NEW = ["london_breakout", "rsi_reversion", "macd_cross", "stochastic_reversion", "gotobi", "pivot_reversal",
       "scalping", "ichimoku", "parabolic_sar", "adx_trend", "perfect_order", "heikin_ashi", "turtle", "carry_trade",
       "long_term_value", "vol_target_trend", "regime_adaptive", "online_logit", "kalman_trend", "bet",
       "martingale", "paroli", "dalembert", "fibonacci", "labouchere", "oscars_grind", "kelly", "nanpin_martingale"]


@pytest.fixture(scope="module")
def market():
    return MARKETS.create("malicious").generate(years=2, seed=3)


@pytest.mark.parametrize("name", NEW)
def test_method_runs_and_trades(market, name):
    strat = STRATEGIES.create(name)
    bars = market.bars(strat.timeframe)
    if name == "long_term_value":                      # 5 年の平均が要るので短い窓で
        strat = STRATEGIES.create(name, {"n": 120, "entry_z": 1.0})
    if name == "carry_trade":                          # 2015〜2016 年の金利差は 1% 未満
        strat = STRATEGIES.create(name, {"min_diff": 0.005})
    bt = Backtester(bars, strat, BrokerConfig.preset("fair"), pip=PIP, seed=1).run()
    assert len(bt.trades) > 0, name
    bt2 = Backtester(bars, STRATEGIES.create(name, strat.info()["params"]), BrokerConfig.preset("fair"),
                     pip=PIP, seed=1).run()
    assert [t.pnl for t in bt.trades] == [t.pnl for t in bt2.trades]


def results(system, seq, **kw):
    s = Staker(system=system, **kw)
    out = []
    for r in seq:
        out.append(s.mult)
        s.update(r)
    return out


def test_staking_systems_follow_their_rules():
    L, W = -1.0, 1.0
    assert results("martingale", [L, L, L, W, L]) == [1, 2, 4, 8, 1]
    assert results("martingale", [L] * 8, max_mult=8) == [1, 2, 4, 8, 1, 2, 4, 8]       # 打ち止めで戻る
    assert results("paroli", [W, W, W, W, L, W]) == [1, 2, 4, 1, 2, 1]
    assert results("dalembert", [L, L, W, W, W]) == [1, 2, 3, 2, 1]
    assert results("fibonacci", [L, L, L, L, W, W]) == [1, 1, 2, 3, 5, 2]
    # 1-2-3 -> 負けで 1-2-3-4 -> 勝ちで 2-3 -> 勝ちで空になり最初から
    assert results("labouchere", [L, W, W, L]) == [4, 5, 5, 4]
    assert results("oscars_grind", [L, L, W, W, W, W]) == [1, 1, 1, 2, 1, 1]


def test_kelly_estimates_the_edge():
    s = Staker(system="kelly", kelly_min_trades=10)
    for r in [2.0, -1.0, 2.0, -1.0, -1.0] * 10:      # 勝率 40%、損益比 2 -> f = 0.4 - 0.6 / 2 = 0.1
        s.update(r)
    assert s.kelly_f() == pytest.approx(0.1)


def test_gotobi_dates():
    d = lambda s: np.array([np.datetime64(s, "ns").astype(np.int64)])  # noqa: E731
    assert is_gotobi(d("2024-08-05"))[0]           # 月曜の 5 日
    assert not is_gotobi(d("2024-08-06"))[0]
    assert is_gotobi(d("2024-08-09"))[0]           # 10 日が土曜 -> 前の金曜
    assert not is_gotobi(d("2024-08-10"))[0]       # 土曜そのものは取引しない
    assert is_gotobi(d("2024-07-31"))[0]           # 月末


class AddScript(Strategy):
    name = "add_script"

    @dataclass
    class Params:
        timeframe: int = 60

    def on_bar(self, i, ctx):
        if i == 0:
            ctx.buy(units=10_000)
        elif i == 2:
            ctx.add(20_000, tp=0.10)


def test_add_averages_the_entry_and_moves_the_target():
    t = trading_times("2021-03-01", days=2)[:600]
    mid = np.r_[np.full(120, 100.0), np.full(120, 99.70), np.full(60, 99.70), np.linspace(99.70, 100.5, 300)]
    bars = make_bars(t, mid, np.full(len(mid), 0.002), 60)
    bt = Backtester(bars, AddScript(), BrokerConfig(slippage_pips=0), pip=PIP).run()
    tr = bt.trades[0]
    assert tr.units == 30_000
    avg = (100.001 * 10_000 + 99.701 * 20_000) / 30_000
    assert tr.entry_price == pytest.approx(avg)
    assert tr.reason == "tp" and tr.exit_price == pytest.approx(avg + 0.10)
