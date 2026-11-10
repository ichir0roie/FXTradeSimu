import numpy as np
import pytest

from fxsim.calendar import NS_HOUR, hour_of_day, trading_times, weekday
from fxsim.data import make_bars
from fxsim.markets import MARKETS
from fxsim.markets.evaluation import compute_facts, score_facts
from fxsim.numerics import _ar1_chunked, ar1_filter


def test_trading_calendar_has_no_weekend():
    t = trading_times("2020-01-01", days=21)
    dow, h = weekday(t), hour_of_day(t)
    assert not np.any(dow == 5)
    assert not np.any((dow == 4) & (h >= 22))
    assert not np.any((dow == 6) & (h < 22))


def test_ar1_matches_loop():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(5000)
    a = 0.995
    ref = np.empty_like(e)
    x = 0.3
    for i, v in enumerate(e):
        x = a * x + v
        ref[i] = x
    np.testing.assert_allclose(ar1_filter(e, a, 0.3), ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(_ar1_chunked(e, a, 0.3), ref, rtol=1e-7, atol=1e-7)


def test_bars_aggregate_ohlc():
    t = trading_times("2021-03-01", days=2)
    rng = np.random.default_rng(1)
    mid = 100 + np.cumsum(rng.standard_normal(len(t))) * 0.01
    spread = np.full(len(t), 0.004)
    b = make_bars(t, mid, spread, 60)
    first = slice(0, b.ticks[0])
    assert b.open[0] == mid[0] and b.close[0] == mid[first][-1]
    assert b.high[0] == mid[first].max() and b.low[0] == mid[first].min()
    np.testing.assert_allclose(b.ask_high - b.high, 0.002)
    # 日足は 22:00 UTC で区切る
    d = make_bars(t, mid, spread, 1440)
    assert np.all(((d.t // NS_HOUR) % 24)[1:] == 22)


@pytest.mark.parametrize("name", ["random_walk", "stylized", "malicious", "real_bridge", "real_bootstrap",
                                  "real_bridge_malicious"])
def test_generation_is_deterministic_and_sane(name):
    m = MARKETS.create(name)
    a = m.generate(days=40, seed=5)
    b = m.generate(days=40, seed=5)
    c = m.generate(days=40, seed=6)
    np.testing.assert_array_equal(a.mid, b.mid)
    assert not np.array_equal(a.mid, c.mid)
    assert np.all(np.isfinite(a.mid)) and np.all(a.spread > 0)
    assert len(a.times) == len(a.mid) == len(a.spread)


def test_real_replay_is_deterministic_and_matches_data():
    from fxsim.data import load_ohlc

    m = MARKETS.create("real")
    assert m.deterministic
    md = m.generate(years=1, seed=1)
    s = load_ohlc("data/real/usdjpy_d1_fred.csv")
    assert md.mid[-1] == pytest.approx(s.close[-1])
    assert md.step_minutes == 1440


def test_real_bridge_passes_through_real_closes():
    m = MARKETS.create("real_bridge")
    md = m.generate(years=0.5, seed=3)
    from fxsim.data import load_ohlc

    s = load_ohlc("data/real/usdjpy_d1_fred.csv").window(years=0.5)
    idx = np.searchsorted(md.times, s.times) - 1
    ok = idx >= 0
    np.testing.assert_allclose(md.mid[idx[ok]], s.close[ok], rtol=1e-9)


def test_component_can_be_disabled_and_typos_rejected():
    full = MARKETS.create("malicious")
    no_hunt = MARKETS.create("malicious", {"stop_hunt": None})
    assert "stop_hunt" in [c.key for c in full.components]
    assert "stop_hunt" not in [c.key for c in no_hunt.components]
    with pytest.raises(KeyError):
        MARKETS.create("malicious", {"stop_hnut": None})
    with pytest.raises(KeyError):
        MARKETS.create("malicious", {"stop_hunt": {"probb": 1}})


def test_malice_events_are_recorded():
    md = MARKETS.create("malicious").generate(days=60, seed=2)
    kinds = {e["kind"] for e in md.events}
    assert {"stop_hunt", "false_breakout", "news"} & kinds


def test_stylized_facts_separate_models():
    rw = compute_facts(MARKETS.create("random_walk").generate(years=1, seed=1))
    st = compute_facts(MARKETS.create("stylized").generate(years=1, seed=1))
    assert abs(rw["kurt_h1"]) < 1.0 and st["kurt_h1"] > 3.0
    assert st["acf_abs_h1_lag1"] > rw["acf_abs_h1_lag1"] + 0.1
    assert score_facts(st)["realism_score"] > score_facts(rw)["realism_score"]
