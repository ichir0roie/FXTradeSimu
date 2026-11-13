"""main_trend_free(S045): 受け取る向きだけにすれば main_carry_breakout と同じ。両方向は払う向きにも入る。"""
from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from tests.test_carry_gate import BASE, _run
from tests.test_carry_trend import _fake_pairs


def test_receive_side_breakout_is_the_carry_breakout(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, a = _run(md, "main_carry_breakout")
    _, b = _run(md, "main_trend_free", {"sides": "recv", "signal": "break"})
    assert len(a.trades) > 0
    assert [(t.pair, t.entry_t, t.pnl) for t in a.trades] == [(t.pair, t.entry_t, t.pnl) for t in b.trades]


def test_both_sides_also_trades_the_paying_side(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    for sides, want in (("both", {1, -1}), ("pay", None)):
        s = STRATEGIES.create("main_trend_free", {**BASE, "sides": sides, "signal": "break"})
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"),
                              EngineConfig(initial_balance=1e6, min_units=1000), seed=1)
        bt.run()
        assert len(bt.trades) > 0
        if want:
            assert {int(t.side) for t in bt.trades} == want
        else:                            # 払う向きだけ: 何日か持った取引のスワップは全部マイナス
            assert all(t.swap < 0 for t in bt.trades if t.bars_held > 5)
