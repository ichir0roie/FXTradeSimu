"""main_carry_hold(S046)・main_carry_short(S047): 上位の数を守る、決めた本数で出る。"""
from fxsim.markets import MARKETS
from tests.test_carry_gate import _run
from tests.test_carry_trend import _fake_pairs


def test_hold_keeps_at_most_top_n_new_pairs(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, one = _run(md, "main_carry_hold", {"top_n": 1, "lt_n": 0, "exit": "now", "emergency_atr": 0.0})
    _, two = _run(md, "main_carry_hold", {"top_n": 2, "lt_n": 0, "exit": "now", "emergency_atr": 0.0})
    assert len(one.trades) > 0 and len(two.trades) > 0
    spans = [(t.entry_t, t.exit_t) for t in one.trades]
    assert all(sum(1 for x, y in spans if x <= a < y) <= 1 for a, _ in spans)
    assert sum(t.bars_held for t in two.trades) >= sum(t.bars_held for t in one.trades)


def test_short_closes_after_the_holding_bars(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, bt = _run(md, "main_carry_short", {"n_out": 0, "hold": 3, "extend": False, "emergency_atr": 0.0})
    assert len(bt.trades) > 0
    assert max(t.bars_held for t in bt.trades[:-1]) <= 3
    _, ext = _run(md, "main_carry_short", {"n_out": 0, "hold": 3, "extend": True, "emergency_atr": 0.0})
    assert max(t.bars_held for t in ext.trades) >= max(t.bars_held for t in bt.trades)
