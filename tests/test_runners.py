"""ランナー(fxsim/runners): 同じ戦略のクラスを、シミュレーションと MT5 の口座で同じ窓口で動かす。"""
import numpy as np
import pytest

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.metrics import compute_metrics
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.markets import MARKETS
from fxsim.runners import Mt5Runner, ReplayRunner, Runner, SimRunner
from fxsim.strategies import STRATEGIES
from fxsim.trading import TraderConfig
from fxsim.trading.portfolio import PortfolioTrader
from fxsim.trading.signals import simulate
from fxsim.trading.trader import LiveTrader


def _md():
    return MARKETS.create("random_walk", {}).generate(years=0.3, seed=3)


def test_sim_runner_matches_a_hand_built_backtest():
    b = load_cases().behaviors["B001"]
    md = _md()
    # random_walk の日付(2015 年〜)は USD − JPY の金利差がほぼ 0 で受け取る向きが無いので、swap_side_only を外す
    r = SimRunner("B001", engine={"swap_side_only": False}).backtest(md, seed=2)
    bt = build_backtester(md, STRATEGIES.create(b.strategy, b.strategy_params()),
                          BrokerConfig.preset(b.broker_spec["preset"], b.broker_spec["overrides"]),
                          r.backtester.cfg, seed=2, schedule=b.trading_schedule())
    bt.run()
    assert r.metrics["n_trades"] == compute_metrics(bt)["n_trades"] > 0
    assert np.allclose(r.backtester.equity_curve, bt.equity_curve, equal_nan=True)


def test_runner_overrides_stack_on_the_behavior():
    r = SimRunner("B047", strategy_params={"n_in": 30}, broker="fair", engine={"initial_balance": 50_000})
    assert r.params["n_in"] == 30 and r.params["n_out"] == 10          # 動作ケースの値に重ねる
    assert r.broker_config().name == "fair" and r.engine_config.initial_balance == 50_000
    assert r.engine_config.swap_side_only and r.multi
    s = SimRunner(strategy="ma_cross", strategy_params={"timeframe": 240})
    assert s.timeframe == 240 and not s.multi and s.behavior is None
    with pytest.raises(ValueError):
        Runner()


def test_decide_is_the_same_core_model_as_the_live_snapshot():
    md = _md()
    r = Runner("B001")
    bars = md.bars(r.timeframe)
    a = r.decide(bars, pip=0.01)
    b = simulate(bars, r.make_strategy(), r.broker_config(), r.engine_config, r.schedule, 0.01)
    assert a.bar_t == b.bar_t and a.orders == b.orders and a.position == b.position


def test_mt5_runner_picks_the_trader_and_turns_off_the_sim_floor():
    one = Mt5Runner(TraderConfig(name="a", behavior="B045", symbol="USDJPY", magic=1))
    assert one.make_strategy().g.protect == 0.0                        # 床は実口座で見る
    assert one.sim_broker.swap_mode == "rates"                          # swap_side_only の動作ケース
    multi = Mt5Runner("configs/trading/demo-majors-b047.yaml")
    assert multi.multi and multi.make_strategy().p.protect == 0.0
    assert multi.config_path.endswith("demo-majors-b047.yaml")

    class _Client:
        pass
    assert isinstance(multi.make_trader(_Client()), PortfolioTrader)
    assert type(one.make_trader(_Client())) is LiveTrader


def test_replay_runner_is_an_mt5_runner_over_the_replay_terminal():
    r = ReplayRunner(["configs/trading/demo-majors-b047.yaml", "configs/replay/majors-b047-loose.yaml"])
    assert isinstance(r, Mt5Runner) and r.multi and len(r.paths) == 2
