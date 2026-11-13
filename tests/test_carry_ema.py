"""main_carry_ema(S044): EMA の向きで入り、返ったら出る。向きは先を見ない。週に 1 回の判定は取引を減らす。"""
import numpy as np

from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from tests.test_carry_gate import BASE, _run
from tests.test_carry_trend import _fake_pairs


def test_it_trades_and_holds_at_least_min_hold_unless_carry_or_stop_ends_it(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    _, daily = _run(md, "main_carry_ema", {"emergency_atr": 0.0})
    _, held = _run(md, "main_carry_ema", {"emergency_atr": 0.0, "min_hold": 5})
    _, weekly = _run(md, "main_carry_ema", {"emergency_atr": 0.0, "signal_every": 5})
    assert len(daily.trades) > 0
    assert min(t.bars_held for t in held.trades[:-1]) >= 5
    assert len(weekly.trades) <= len(daily.trades)


def test_the_ema_direction_does_not_look_ahead(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    legs = md.meta["legs"]
    full = STRATEGIES.create("main_carry_ema", BASE)
    full.prepare({p: leg.bars(1440) for p, leg in legs.items()}, np.random.default_rng(0))
    n = len(full.t)
    part = STRATEGIES.create("main_carry_ema", BASE)
    part.prepare({p: leg.bars(1440).slice(n - 50) for p, leg in legs.items()}, np.random.default_rng(0))
    for p in legs:
        assert np.array_equal(full.trend[p][:n - 50], part.trend[p])
