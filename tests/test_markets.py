import numpy as np
import pytest

from fxsim.calendar import NS_HOUR, hour_of_day, trading_times, weekday
from fxsim.data import load_ohlc, make_bars
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
    assert abs(rw["kurt_h1"]) < 1.0 and st["kurt_h1_trim"] > 3.0
    assert st["acf_abs_h1_lag1"] > rw["acf_abs_h1_lag1"] + 0.1
    assert score_facts(st)["realism_score"] > score_facts(rw)["realism_score"]


def test_placebo_keeps_real_magnitudes_but_not_direction():
    from fxsim.markets.components.real import _placebo

    s = load_ohlc("data/real/usdjpy_d1_fred.csv").window(years=2)
    lc = np.log(s.close)
    c, o, h, lw = _placebo(lc, np.log(s.open), np.log(s.high), np.log(s.low), np.random.default_rng(0))
    np.testing.assert_allclose(np.abs(np.diff(c)), np.abs(np.diff(lc)), atol=1e-12)
    assert np.mean(np.sign(np.diff(c)) == np.sign(np.diff(lc))) < 0.6
    assert np.all(h >= np.maximum(o, c) - 1e-12) and np.all(lw <= np.minimum(o, c) + 1e-12)


def test_bridge_v2_has_no_intraday_mean_reversion():
    """v1 は 24 時間の分散比が 0.9 を割っていた(R005)。v2 は 1 に近い。

    完全に戻るフラッシュクラッシュ(v3)は本物の平均回帰で、1 回で分散比を大きく下げるので外して測る。
    """
    vrs = []
    for seed in (1, 2, 3):
        md = MARKETS.create("real_bridge", {"flash_overlay": None}).generate(years=1, seed=seed)
        c = np.log(md.bars(60).close)
        vrs.append(np.var(c[24:] - c[:-24]) / (24 * np.var(np.diff(c))))
    assert 0.93 < np.mean(vrs) < 1.07


_SIMD_SCRIPT = """
import json, sys
from fxsim import catalog  # noqa: F401
from fxsim.markets import MARKETS
out = {}
for model in ("malicious", "real_bridge_malicious"):
    md = MARKETS.create(model, {}).generate(years=1, seed=3)
    out[model] = [len(md.events), [e["kind"] for e in md.events][:2000], float(md.mid[-1])]
json.dump(out, sys.stdout)
"""


def test_markets_do_not_depend_on_the_cpu_simd_features():
    # numpy は CPU の SIMD(AVX2 / AVX-512)で並べ替えや exp の実装を切り替える。同じ値の並べ替えの順番に
    # 依存すると、同じシードでも端末ごとに別の相場になる(NewsShocks の argsort で起きていた)
    import json
    import os
    import subprocess
    import sys

    def run(env_extra):
        env = {**os.environ, **env_extra}
        res = subprocess.run([sys.executable, "-c", _SIMD_SCRIPT], capture_output=True, text=True, env=env,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
        return json.loads(res.stdout)

    a = run({})
    b = run({"NPY_DISABLE_CPU_FEATURES": "X86_V3 X86_V4 AVX512F AVX512_SKX AVX512_ICL AVX512_SPR AVX2 FMA3"})
    for model in a:
        assert a[model][0] == b[model][0] and a[model][1] == b[model][1], model
        assert a[model][2] == pytest.approx(b[model][2], rel=1e-9), model


def _rv_leak_ratio(md) -> float:
    """日の前半の実現分散(直前 20 日比)で上位 2 割の日と下位 2 割の日の、E[r²] / E[実現分散] の比。

    足の中の荒れ方から日足のリターンの大きさが漏れていなければ 1 に近い(R008)。
    """
    from fxsim.calendar import trading_day

    lp = np.log(md.mid)
    dx = np.diff(lp, prepend=lp[0])
    d = trading_day(md.times)
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    ends = np.r_[starts[1:], len(dx)]
    rv = np.add.reduceat(dx * dx, starts)
    ret = np.add.reduceat(dx, starts)
    half = np.array([(dx[a:(a + b) // 2] ** 2).sum() for a, b in zip(starts, ends, strict=True)])
    prev = np.convolve(rv, np.ones(20) / 20, mode="full")[:len(rv)]
    key = (half / np.r_[np.nan, prev[:-1]])[20:]
    ret, rv = ret[20:], rv[20:]
    lo, hi = key <= np.quantile(key, 0.2), key >= np.quantile(key, 0.8)
    return float((np.mean(ret[hi] ** 2) / np.mean(rv[hi])) / (np.mean(ret[lo] ** 2) / np.mean(rv[lo])))


def test_bridge_v4_does_not_leak_daily_return_through_intraday_volatility():
    """v3(day_weight 0.6)は静かな日に戻り・荒れた日に走り、比が 2.5 を超えていた。v4 は 1 に近い。"""
    v4 = [_rv_leak_ratio(MARKETS.create("real_bridge_placebo", {"flash_overlay": None}).generate(years=4, seed=s))
          for s in (1, 2)]
    v3 = _rv_leak_ratio(MARKETS.create("real_bridge_placebo", {"flash_overlay": None, "anchor": {
        "calibrate_vol": "local", "clock": "variance"}}).generate(years=4, seed=1))
    assert 0.7 < np.mean(v4) < 1.4
    assert v3 > 2.0


def test_quick_moves_finish_within_their_steps():
    from fxsim.markets.base import GenContext
    from fxsim.markets.blocks import apply_quick_moves

    ctx = GenContext(rng=np.random.default_rng(0), times=np.arange(30), base=None, vol=np.zeros(30),
                     drift=np.zeros(30), jumps=np.zeros(30))
    ctx.logp = np.zeros(30)
    apply_quick_moves(ctx, [(10, 1, 0.01), (20, 3, -0.03)])
    assert ctx.logp[9] == 0 and ctx.logp[10] == pytest.approx(0.01) and ctx.logp[19] == pytest.approx(0.01)
    assert ctx.logp[22] == pytest.approx(-0.02) and ctx.logp[-1] == pytest.approx(-0.02)
