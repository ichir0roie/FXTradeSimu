"""本命 2 本目(main_carry_trend、S039)と、共通の対策のスプレッドのゲート・mid の終値の損切り(guards)。"""
import numpy as np
import pytest

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.calendar import NS_DAY, server_hour
from fxsim.data import load_rates
from fxsim.data.market import MarketData
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from fxsim.strategies.guards import GuardConfig, spread_ok, typical_spread


def test_typical_spread_is_causal_and_ignores_a_single_spike():
    s = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0])
    t = typical_spread(s, 3)
    assert t.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    s2 = s.copy()
    s2[5] = 100.0                                  # 後の足を変えても、前の足の典型は変わらない
    assert typical_spread(s2, 3)[:5].tolist() == t[:5].tolist()


def test_spread_ok():
    g = GuardConfig.from_dict({"max_spread_mult": 3.0})
    assert spread_ok(2.9, 1.0, g) and not spread_ok(3.1, 1.0, g)
    assert spread_ok(100.0, 1.0, GuardConfig())    # 0 なら見ない
    assert spread_ok(5.0, float("nan"), g)          # 典型が分からなければ止めない
    with pytest.raises(ValueError):
        GuardConfig.from_dict({"stop_price": "last"})


def test_server_hour_puts_the_rollover_at_midnight_in_summer_and_winter():
    summer = np.datetime64("2026-07-15T21:00", "ns").astype(np.int64)    # NY 17 時(夏時間)
    winter = np.datetime64("2026-12-15T22:00", "ns").astype(np.int64)    # NY 17 時(冬時間)
    assert server_hour(np.array([summer, winter]), "America/New_York+7").tolist() == [0, 0]


def _daily(close, pair="usdjpy"):
    n = len(close)
    t0 = np.datetime64("2019-01-07T22:00", "ns").astype(np.int64)
    times = t0 + np.arange(n, dtype=np.int64) * NS_DAY
    md = MarketData(times=times, mid=np.asarray(close, float), spread=np.full(n, 0.004), pip=0.01,
                    step_minutes=1440, meta={"swap_pair": pair})
    return md


def test_carry_trend_only_holds_the_receiving_side_and_needs_the_trend():
    close = np.r_[np.linspace(100, 110, 60), np.linspace(110, 95, 60)]
    md = _daily(close)
    b = md.bars(1440)
    diff = load_rates(MULTI_RATES).pair_differential("usdjpy", b.t)
    s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 0})
    target, stop, move = s.signals(b, np.random.default_rng(1))
    assert np.array_equal(target, np.sign(diff).astype(int))            # 金利差の符号の向きだけ
    s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 10})
    target, _, _ = s.signals(b, np.random.default_rng(1))
    side = int(np.sign(diff[0]))
    ma = np.convolve(b.close, np.ones(10) / 10, "full")[:len(b)]
    for i in range(10, len(b)):
        agree = np.sign(b.close[i] - ma[i]) == side
        assert target[i] == (side if agree else 0)
    s = STRATEGIES.create("main_carry_trend", {"min_carry": 1.0})      # 年 100% の金利差は無い
    assert not s.signals(b, np.random.default_rng(1))[0].any()


def test_spread_gate_blocks_new_entries_when_the_spread_is_wide():
    md = _daily(np.linspace(100, 130, 120))
    results = {}
    for mult in (0.0, 0.5):     # 0.5 倍: 今のスプレッドは必ず典型の 0.5 倍を超える
        s = STRATEGIES.create("main_carry_trend", {"min_carry": -1.0, "trend_n": 0,
                                                   "guard": {"max_spread_mult": mult, "cost_multiple": 0}})
        bt = build_backtester(md, s, BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}}),
                              EngineConfig(), seed=1)
        bt.run()
        results[mult] = (len(bt.deals), s.blocked["spread"])
    assert results[0.0][0] > 0 and results[0.0][1] == 0
    assert results[0.5][0] == 0 and results[0.5][1] > 0


class _Always:
    """毎足、同じ向きに入る(1 ペアのエンジン用)。"""
    multi = False
    warmup = 0
    timeframe = 1440

    def __init__(self, side):
        self.side = side

    def prepare(self, bars, rng):
        pass

    def on_bar(self, i, ctx):
        if ctx.side != self.side:
            (ctx.buy if self.side > 0 else ctx.sell)(units=10_000)


def test_swap_side_only_blocks_entries_that_pay_swap_in_every_engine():
    md = _daily(np.linspace(130, 100, 120))           # 下げ続ける: main_trend_d1 は売りたがる
    b = md.bars(1440)
    diff = load_rates(MULTI_RATES).pair_differential("usdjpy", b.t)
    assert (diff > 0.0044).all()                      # この期間の USDJPY は買いがスワップを受け取る
    broker = BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}})
    for side, want_trades in [(1, True), (-1, False)]:
        bt = build_backtester(md, _Always(side), broker, EngineConfig(swap_side_only=True), seed=1)
        bt.run()
        assert (len(bt.trades) > 0) == want_trades
        assert (bt.stats["swap_blocked"] > 0) == (not want_trades)
    # MT5 のテスター(本命の戦略)
    s = STRATEGIES.create("main_trend_d1", {"slow": 20, "mom": 10, "guard": {"cost_multiple": 0}})
    bt = build_backtester(md, s, broker, EngineConfig(swap_side_only=True), seed=1)
    bt.run()
    assert len(bt.deals) == 0 and bt.stats["swap_blocked"] > 0
    s = STRATEGIES.create("main_trend_d1", {"slow": 20, "mom": 10, "guard": {"cost_multiple": 0}})
    bt = build_backtester(md, s, broker, EngineConfig(), seed=1)
    bt.run()
    assert len(bt.deals) > 0


def _fake_pairs(tmp_path, monkeypatch, n=400):
    from fxsim import paths
    from fxsim.data import load_ohlc
    from fxsim.markets.models import real_pairs
    from fxsim.storage import prices
    path = tmp_path / "market.sqlite"
    monkeypatch.setattr(paths, "MARKET_DB", path)
    load_ohlc.cache_clear()
    real_pairs._aligned.cache_clear()
    con = prices.open_market_db(path)
    t0 = np.datetime64("2010-01-04", "s").astype(np.int64)
    rng = np.random.default_rng(3)
    for sym, p0, digits in (("USDJPY", 100.0, 3), ("AUDJPY", 80.0, 3)):
        c = p0 * np.exp(np.cumsum(rng.normal(0.0004, 0.006, n)))
        o = np.r_[c[0], c[:-1]]
        t = t0 + np.arange(n, dtype=np.int64) * 86_400
        prices.upsert_symbol(con, "Test", sym, digits, 10.0 ** -digits, "UTC+2")
        prices.write_rates(con, "Test", sym, 1440, {
            "open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
            "tick_volume": np.full(n, 10), "spread": np.full(n, 10), "real_volume": np.zeros(n, np.int64),
            "server_time": t + 7200, "time": t}, "UTC+2")
    con.commit()
    con.close()


def test_real_pairs_bootstrap_carries_rates_with_their_dates(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    params = {"pairs": ["USDJPY", "AUDJPY"], "source": {"start": "2010-06-01", "end": "2010-12-31", "block_days": 20}}
    real = MARKETS.create("real_pairs", params).generate(seed=1)
    a = MARKETS.create("real_pairs_bootstrap", params).generate(seed=1)
    b = MARKETS.create("real_pairs_bootstrap", params).generate(seed=2)
    ra, rb = real.meta["legs"]["audjpy"].bars(1440), a.meta["legs"]["audjpy"].bars(1440)
    assert len(ra) == len(rb) and np.array_equal(ra.t, rb.t)
    assert ra.rate_diff is None and rb.rate_diff is not None
    assert not np.allclose(rb.close, b.meta["legs"]["audjpy"].bars(1440).close)       # 乱数で別の歴史
    assert rb.close[0] == pytest.approx(ra.close[0])                                    # 水準は置く先から始まる
    assert np.allclose(rb.spread_mean, ra.spread_mean)                                  # スプレッドは置く先のまま
    # 全ペアで同じ日付の並び: 金利差の日付が、USDJPY と AUDJPY で同じ並び
    ua = a.meta["legs"]["usdjpy"].bars(1440).rate_diff
    rates = load_rates(MULTI_RATES)
    pool_t = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1).meta["legs"]["usdjpy"] \
        .bars(1440).t
    usd = rates.pair_differential("usdjpy", pool_t)
    aud = rates.pair_differential("audjpy", pool_t)
    for k in range(1, len(ua), 37):
        j = np.flatnonzero(np.isclose(usd, ua[k]) & np.isclose(aud, rb.rate_diff[k]))
        assert len(j) > 0


def test_carry_breakout_enters_only_on_breakouts_in_the_carry_direction(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    md = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1)
    s = STRATEGIES.create("main_carry_breakout", {"n_in": 20, "n_out": 10, "min_carry": -1.0, "c2r_min": 0.0,
                                                 "vol_n": 20, "protect": 0.0, "max_spread_mult": 0.0})
    bt = build_backtester(md, s, BrokerConfig.preset("oanda"), EngineConfig(initial_balance=1e5, min_units=1000,
                                                                            swap_side_only=True), seed=1)
    bt.run()
    assert len(bt.trades) > 0
    legs = md.meta["legs"]
    for tr in bt.trades:
        b = legs[tr.pair].bars(1440)
        k = int(np.searchsorted(b.t, tr.entry_t, "right")) - 2                     # 入る判断をした足
        side = int(np.sign(load_rates(MULTI_RATES).pair_differential(tr.pair, b.t[k:k + 1])[0]))
        assert tr.side == side                                                     # スワップを受け取る向き
        hi, lo = b.high[k - 20:k].max(), b.low[k - 20:k].min()
        assert (b.close[k] > hi) if side > 0 else (b.close[k] < lo)               # ブレイクアウト


def test_carry_rank_ranks_stable_pairs_by_carry_to_risk_and_sizes_by_weight():
    s = STRATEGIES.create("main_carry_rank", {"weights": [4, 3, 2, 1], "unit": 1000, "min_carry": 0.01,
                                              "max_vol": 0.15, "max_spread_days": 10})
    s.pairs = ["a", "b", "c", "d", "e", "f", "g"]
    one = lambda v: np.array([v], dtype=float)  # noqa: E731
    #            carry  vol   spread_days
    spec = {"a": (0.03, 0.10, 1.0),    # c2r 0.30 -> 2 位
            "b": (0.02, 0.05, 1.0),    # c2r 0.40 -> 1 位
            "c": (0.04, 0.20, 1.0),    # ボラが大きすぎる -> 外す
            "d": (0.005, 0.02, 1.0),   # 金利差が下限より小さい -> 外す
            "e": (0.02, 0.10, 30.0),   # スプレッドを取り返すのに 30 日 -> 外す
            "f": (0.015, 0.10, 2.0),   # c2r 0.15 -> 4 位
            "g": (0.02, 0.08, 2.0)}    # c2r 0.25 -> 3 位
    s.carry = {k: one(v[0]) for k, v in spec.items()}
    s.vol_rank = {k: one(v[1]) for k, v in spec.items()}
    s.spread_days = {k: one(v[2]) for k, v in spec.items()}
    s.typical = {k: one(1.0) for k in spec}
    s.side = {k: np.array([1]) for k in spec}
    s.rank_hist = {k: 0 for k in spec}
    s.rank_bar(0)
    assert s.rank == {"b": 0, "a": 1, "g": 2, "f": 3}
    assert [s.entry_units(k, 0) for k in ("b", "a", "g", "f")] == [4000, 3000, 2000, 1000]
    assert not any(s.allowed(k, 0) for k in ("c", "d", "e"))
    s.p.weights = [4, 3]
    assert s.allowed("a", 0) and not s.allowed("g", 0)
    s.p.drop_rank = 2
    assert s.drop("g", 0) and s.drop("c", 0) and not s.drop("a", 0)


def test_carry_breakout_shock_exit_reads_closes_and_volatility():
    from fxsim.strategies.main.carry_breakout import _rolling_std
    r = np.random.default_rng(1).normal(0, 0.01, 40)
    got = _rolling_std(r, 5)
    assert np.isnan(got[3]) and got[10] == pytest.approx(r[6:11].std(ddof=1))
    s = STRATEGIES.create("main_carry_breakout", {"shock_atr": 2.5, "vol_ratio": 2.0})
    s.c = {"a": np.array([100.0, 100.0, 97.0, 98.0])}
    s.atr = {"a": np.array([1.0, 1.0, 1.0, 1.0])}
    s.vol_fast = {"a": np.array([0.1, 0.1, 0.1, 0.3])}
    s.vol_slow = {"a": np.array([0.1, 0.1, 0.1, 0.1])}
    assert not s._shock("a", 1, 1)
    assert s._shock("a", 2, 1)                 # 1 日で 3 = ATR x 3 の逆行
    assert not s._shock("a", 2, -1)            # 売りには順行
    assert s._shock("a", 3, -1)                # 逆行は 1 だけだが、5 日ボラが 250 日の 3 倍


def test_real_pairs_placebo_flips_each_day_for_all_pairs_and_keeps_sizes(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    params = {"pairs": ["USDJPY", "AUDJPY"], "source": {"start": "2010-06-01", "end": "2010-12-31"}}
    real = MARKETS.create("real_pairs", params).generate(seed=1)
    pl = MARKETS.create("real_pairs_placebo", params).generate(seed=1)
    rs = {}
    for p in ("usdjpy", "audjpy"):
        a, b = real.meta["legs"][p].bars(1440), pl.meta["legs"][p].bars(1440)
        assert np.array_equal(a.t, b.t) and np.allclose(a.spread_mean, b.spread_mean)
        ra, rb = np.diff(np.log(a.close)), np.diff(np.log(b.close))
        assert np.allclose(np.abs(ra), np.abs(rb))                                     # 大きさはそのまま
        rs[p] = np.sign(ra) * np.sign(rb)
        assert (rs[p] < 0).any() and (rs[p] > 0).any()                                 # 向きはでたらめ
    nz = (rs["usdjpy"] != 0) & (rs["audjpy"] != 0)
    assert np.array_equal(rs["usdjpy"][nz], rs["audjpy"][nz])                          # 全ペアで同じ符号


def test_real_pairs_bootstrap_demean_removes_the_pool_drift(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch)
    src = {"start": "2010-06-01", "end": "2010-12-31", "block_days": 20}
    pool = MARKETS.create("real_pairs", {"pairs": ["USDJPY", "AUDJPY"]}).generate(seed=1).meta["legs"]["usdjpy"]
    mu = float(np.mean(np.diff(np.log(pool.bars(1440).close))))
    assert mu > 0                                                                      # 偽の相場は上げのドリフト
    for seed in (1, 2):
        a = MARKETS.create("real_pairs_bootstrap", {"pairs": ["USDJPY", "AUDJPY"], "source": src}) \
            .generate(seed=seed).meta["legs"]["usdjpy"].bars(1440)
        d = MARKETS.create("real_pairs_bootstrap", {"pairs": ["USDJPY", "AUDJPY"], "source": {**src, "demean": True}}) \
            .generate(seed=seed).meta["legs"]["usdjpy"].bars(1440)
        ra, rd = np.diff(np.log(a.close)), np.diff(np.log(d.close))
        assert np.allclose(ra - rd, mu)                                                # 1 本ごとに同じだけ引く
        assert np.all(d.low <= d.close + 1e-9) and np.all(d.high >= d.close - 1e-9)


def test_verification_of_a_real_pairs_behavior_uses_real_pairs_patterns_at_two():
    from fxsim.cases import verification_case
    from fxsim.cases.loader import load_cases
    from fxsim.cases.runner import to_experiment
    from fxsim.verification import GROUPS, REAL_PAIRS_FILL_HOUR, REAL_PAIRS_PATTERNS

    t = verification_case("B049", years=30, seeds=[1, 2], block_days=250, start="2000-01-01", end="2026-01-01")
    assert [p.market for p in t.patterns] == REAL_PAIRS_PATTERNS + ["real_pairs_bootstrap"]
    assert all(p.params["source"]["daily_fill_hour"] == REAL_PAIRS_FILL_HOUR for p in t.patterns)
    ref = t.patterns[-1]
    assert ref.display_label == "real_pairs_bootstrap_demeaned" and ref.params["source"]["demean"] is True
    assert [GROUPS[p.display_label] for p in t.patterns] == ["real", "bootstrap", "placebo", "demeaned"]
    cfg = to_experiment(t, load_cases().behaviors["B049"])
    assert [s.name for s in cfg.strategies] == ["main_carry_breakout", "basket_timing_null"]
    assert all(m.params["source"]["start"] == "2000-01-01" for m in cfg.markets)


def test_timing_null_places_the_same_holdings_on_allowed_bars_without_overlap():
    from fxsim.research.carry_timing_null import place

    n = 400
    ok = {"usdjpy": np.zeros(n, bool), "audjpy": np.ones(n, bool)}
    ok["usdjpy"][100:300] = True
    lengths = {"usdjpy": [10, 20, 5], "audjpy": [30, 30]}
    plan, missed = place(lengths, ok, warmup=50, n=n, rng=np.random.default_rng(1))
    assert missed == 0
    for pair, trades in plan.items():
        assert sorted(L for _, L in trades) == sorted(lengths[pair])        # 同じ本数の束
        assert all(ok[pair][d] and d >= 50 for d, _ in trades)              # 入りは許された足だけ
        spans = sorted((d, d + L + 1) for d, L in trades)
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:], strict=False))           # 同じペアで重ならない


def test_timing_null_keeps_the_inner_holdings_and_only_moves_them_in_time(tmp_path, monkeypatch):
    """時期だけでたらめ版: 中の戦略と同じペア・同じ持つ本数・同じ量の取引を、受け取る向きで、別の時期に建てる。"""
    _fake_pairs(tmp_path, monkeypatch, n=800)
    params = {"pairs": ["USDJPY", "AUDJPY"], "source": {"start": "2010-01-01", "end": "2012-03-01"}}
    md = MARKETS.create("real_pairs", params).generate(seed=1)
    inner = {"n_in": 10, "n_out": 5, "min_carry": 0.005, "c2r_min": 0.0, "units": 2000, "protect": 0.0}
    cfg = EngineConfig(initial_balance=100_000, min_units=1000, swap_side_only=True)
    broker = BrokerConfig.preset("oanda")
    main = build_backtester(md, STRATEGIES.create("main_carry_breakout", inner), broker, cfg, seed=1)
    main.run()
    assert len(main.trades) >= 5
    null = STRATEGIES.create("basket_timing_null", {"inner": "main_carry_breakout", "inner_params": inner})
    bt = build_backtester(md, null, broker, cfg, seed=1)
    bt.run()

    def book(trades):
        return sorted((t.pair, t.bars_held, t.units) for t in trades)
    assert book(bt.trades) == book(main.trades)
    assert sorted(t.entry_t for t in bt.trades) != sorted(t.entry_t for t in main.trades)
    for t in bt.trades:                                                     # 受け取る向きだけ
        leg = bt.legs[t.pair]
        k = int(np.searchsorted(leg.t, t.entry_t))
        assert leg.swap_side_ok(t.side, k - 1)
    assert null.stats["entered"] == len(main.trades) and null.stats["max_open"] <= null.max_open


def test_carry_breakout_take_profit_by_atr_and_reentry_keep_the_default_unchanged(tmp_path, monkeypatch):
    _fake_pairs(tmp_path, monkeypatch, n=800)
    params = {"pairs": ["USDJPY", "AUDJPY"], "source": {"start": "2010-01-01", "end": "2012-03-01"}}
    md = MARKETS.create("real_pairs", params).generate(seed=1)
    base = {"n_in": 10, "n_out": 5, "min_carry": 0.005, "c2r_min": 0.0, "units": 2000, "protect": 0.0}
    cfg = EngineConfig(initial_balance=100_000, min_units=1000, swap_side_only=True)
    broker = BrokerConfig.preset("oanda")
    runs = {}
    for name, extra in (("base", {}), ("off", {"tp_atr": 0.0, "reenter": True}), ("tp", {"tp_atr": 0.5}),
                        ("re", {"tp_atr": 0.5, "reenter": True})):
        s = STRATEGIES.create("main_carry_breakout", {**base, **extra})
        bt = build_backtester(md, s, broker, cfg, seed=1)
        bt.run()
        runs[name] = (s, bt)
    assert runs["off"][1].equity_curve[-1] == runs["base"][1].equity_curve[-1]      # 利確なしなら入り直しは効かない
    s, bt = runs["tp"]
    assert s.tp_count > 0 and len(bt.trades) > len(runs["base"][1].trades)
    s2, bt2 = runs["re"]
    assert s2.tp_count >= s.tp_count and len(bt2.trades) > len(bt.trades)           # 入り直しで刻む回数が増える


def test_carry_breakout_take_profit_order_sits_on_the_server(tmp_path, monkeypatch):
    """tp_order: 利確を建値 ± ATR x tp_atr の指値で置き、足の中で触れたら閉じる。閉じた分を数えて入り直す(E041)。"""
    _fake_pairs(tmp_path, monkeypatch, n=800)
    params = {"pairs": ["USDJPY", "AUDJPY"], "source": {"start": "2010-01-01", "end": "2012-03-01"}}
    md = MARKETS.create("real_pairs", params).generate(seed=1)
    base = {"n_in": 10, "n_out": 5, "min_carry": 0.005, "c2r_min": 0.0, "units": 2000, "protect": 0.0}
    cfg = EngineConfig(initial_balance=100_000, min_units=1000, swap_side_only=True)
    broker = BrokerConfig.preset("oanda")
    runs = {}
    for name, extra in (("base", {}), ("order", {"tp_atr": 0.5, "tp_order": True}),
                        ("re", {"tp_atr": 0.5, "tp_order": True, "reenter": True})):
        s = STRATEGIES.create("main_carry_breakout", {**base, **extra})
        bt = build_backtester(md, s, broker, cfg, seed=1)
        bt.run()
        runs[name] = (s, bt)
    s, bt = runs["order"]
    tps = [t for t in bt.trades if t.reason == "tp"]
    assert tps and s.tp_count == len(tps)                       # サーバーの指値で閉じ、戦略がそれを数える
    for t in tps:                                               # 指値の値段 = 建値 ± ATR(建てた判断の足)x 0.5
        assert (t.exit_price - t.entry_price) * t.side > 0
    s2, bt2 = runs["re"]
    assert s2.tp_count >= s.tp_count and len(bt2.trades) > len(bt.trades)
    assert all(t.reason != "tp" for t in runs["base"][1].trades)
