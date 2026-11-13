"""異常な値動き(ExtremeEvents / FlashCrashOverlay)と薄商い(HolidayLiquidity)。"""
import numpy as np

from fxsim.calendar import day_of_month
from fxsim.markets import MARKETS

MANY = {"flash_per_year": 20.0, "intervention_per_year": 10.0, "shock_per_year": 5.0}


def _kinds(md) -> dict:
    out: dict[str, list] = {}
    for e in md.events:
        out.setdefault(e["kind"], []).append(e)
    return out


def test_extreme_events_happen_widen_spread_and_are_deterministic():
    a = MARKETS.create("stylized", {"extreme": MANY}).generate(years=1, seed=3)
    b = MARKETS.create("stylized", {"extreme": MANY}).generate(years=1, seed=3)
    np.testing.assert_array_equal(a.mid, b.mid)
    ev = _kinds(a)
    assert {"flash_crash", "intervention", "shock"} <= set(ev)
    pip = a.pip
    for e in ev["flash_crash"]:
        i = e["idx"]
        assert np.median(a.spread[i:i + 3]) / pip > 3 * np.median(a.spread) / pip
    # フラッシュクラッシュは段で動く: 1 本で数十 pip 飛ぶ足がある
    i = ev["flash_crash"][0]["idx"]
    jumps = np.abs(np.diff(a.mid[i:i + 15])) / pip
    assert jumps.max() > 20


def test_intervention_goes_against_the_recent_trend():
    only = {"flash_per_year": 0.0, "shock_per_year": 0.0, "intervention_per_year": 20.0}
    md = MARKETS.create("stylized", {"extreme": only}).generate(years=2, seed=5)
    base = MARKETS.create("stylized", {"extreme": None}).generate(years=2, seed=5)   # 介入の前の経路
    ev = _kinds(md)["intervention"]
    assert len(ev) >= 10
    lb = 60 * 1440 // md.step_minutes
    trend = [np.sign(base.mid[e["idx"]] - base.mid[max(0, e["idx"] - lb)]) for e in ev]
    assert all(t == -e["direction"] for t, e in zip(trend, ev, strict=True))


def test_flash_overlay_keeps_daily_closes_of_the_real_skeleton():
    base = MARKETS.create("real_bridge", {"flash_overlay": None}).generate(years=2, seed=2)
    hit = MARKETS.create("real_bridge", {"flash_overlay": {"flash_per_year": 40.0}}).generate(years=2, seed=2)
    assert len(_kinds(hit).get("flash_crash", [])) >= 20
    idx = hit.meta["anchor"]["idx"]
    np.testing.assert_allclose(hit.mid[idx], base.mid[idx], rtol=1e-12)
    np.testing.assert_allclose(hit.bars(1440).close, base.bars(1440).close, rtol=1e-12)
    assert hit.bars(1440).low.min() < base.bars(1440).low.min() or \
        np.any(hit.bars(1440).low < base.bars(1440).low - 0.2)


def test_thin_liquidity_widens_spread_around_new_year():
    md = MARKETS.create("stylized", {"extreme": None}).generate(years=1, seed=1)
    thin = md.meta["thin_mask"]
    d = day_of_month(md.times)
    assert thin[(d == 27)].any() and not thin[(d == 15)].any()
    assert np.median(md.spread[thin]) > 1.8 * np.median(md.spread[~thin])
