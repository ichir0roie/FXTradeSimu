"""main_pullback_d1(S048): 長いトレンドの向きにだけ、RSI(2) の行き過ぎで入り、戻ったら出る。"""
import numpy as np

from fxsim.strategies import STRATEGIES


class _Bars:
    def __init__(self, close):
        self.close = np.asarray(close, dtype=float)
        self.high = self.close + 0.2
        self.low = self.close - 0.2
        self.open = self.close
        self.timeframe = 1440
        self.spread_mean = np.full(len(self.close), 0.01)

    def __len__(self):
        return len(self.close)


def _target(close, **params):
    s = STRATEGIES.create("main_pullback_d1", {"slow": 20, **params})
    target, stop, move = s.signals(_Bars(close), np.random.default_rng(0))
    return np.asarray(target), stop, move


def test_buys_a_dip_only_in_an_uptrend_and_leaves_when_rsi_recovers():
    up = np.r_[np.linspace(100, 120, 60), 119.5, 119.0, 120.5, 121.5, 122.0]      # 上げの中の 2 日の押し
    t, stop, _ = _target(up, enter=30.0)
    assert t[:60].tolist() == [0] * 60 and t[61] == 1           # 押した足の終値で買い
    assert t[-1] == 0                                           # RSI(2) が 50 を越えたら出る
    assert (stop[30:] > 0).all()
    down = 220 - up                                             # 鏡: 下げの中の戻りで売り
    assert _target(down, enter=30.0)[0].tolist() == (-t).tolist()


def test_does_not_fade_against_the_trend_unless_asked():
    crash = np.r_[np.linspace(100, 120, 60), np.linspace(118, 90, 15)]             # 平均を割って下げ続ける
    t, _, _ = _target(crash)
    below = np.flatnonzero(crash < np.convolve(crash, np.ones(20) / 20, "full")[:len(crash)])
    assert (t[below[below >= 20]] <= 0).all()                   # 平均の下では買わない
    assert (_target(crash, trend=False)[0][below[below >= 20]] > 0).any()          # 対照(ただの逆張り)は買う


def test_time_exit():
    slide = np.r_[np.linspace(100, 140, 60), 139.0, np.linspace(138.9, 138.0, 20)]  # 押したまま戻らない
    t, _, _ = _target(slide, max_bars=5)
    run = np.flatnonzero(t == 1)
    assert len(run) and (np.diff(run) == 1).sum() <= len(run) and t[60 + 5] == 0


def test_no_lookahead():
    from fxsim.backtest.lookahead import check_lookahead
    from fxsim.markets import MARKETS

    md = MARKETS.create("malicious").generate(years=2, seed=11)
    assert check_lookahead(lambda: STRATEGIES.create("main_pullback_d1"), md.bars(1440)) == []


# ---- 指値で待つ版(S049) ---------------------------------------------------------

def _hourly(seed=3, years=3):
    from fxsim.markets import MARKETS

    return MARKETS.create("malicious").generate(years=years, seed=seed)


def _run_limit(md, params=None, **engine):
    from fxsim.backtest.broker import BrokerConfig
    from fxsim.backtest.engine import EngineConfig
    from fxsim.backtest.portfolio import build_backtester

    s = STRATEGIES.create("main_pullback_limit", {"slow": 50, **(params or {})})
    bt = build_backtester(md, s, BrokerConfig.preset("oanda"), EngineConfig(initial_balance=1e6, **engine), seed=1)
    bt.run()
    return s, bt


def test_limit_version_has_no_lookahead():
    from fxsim.backtest.lookahead import check_lookahead

    md = _hourly(years=1)
    assert check_lookahead(lambda: STRATEGIES.create("main_pullback_limit", {"slow": 50}), md.bars(60)) == []


def test_limit_entries_fill_at_the_limit_or_better_and_only_some_orders_fill():
    md = _hourly()
    s, bt = _run_limit(md)
    assert s.stats["limit"] > len(bt.trades) > 0           # 届かなかった指値は取り消される
    m, bm = _run_limit(md, {"limit_atr": 0.0})
    assert m.stats["limit"] == 0 and m.stats["market"] == len(bm.trades) > len(bt.trades)


def test_swap_side_rule_also_blocks_pending_orders():
    md = _hourly()
    _, both = _run_limit(md)
    _, recv = _run_limit(md, swap_side_only=True)
    assert recv.stats["swap_blocked"] > 0
    assert len({int(t.side) for t in recv.trades}) <= 1 < len({int(t.side) for t in both.trades})


def test_volatility_gate_only_removes_entries():
    md = _hourly()
    _, free = _run_limit(md, {"limit_atr": 0.0})
    _, calm = _run_limit(md, {"limit_atr": 0.0, "vol_ratio_min": 1.1, "vol_long": 100})
    assert 0 < len(calm.trades) < len(free.trades)


# ---- 日足未満の版(signal_hours < 24、E052) -------------------------------------------

def test_intraday_version_has_no_lookahead_and_decides_on_four_hour_closes():
    from fxsim.backtest.lookahead import check_lookahead

    md = _hourly(years=1)
    params = {"slow": 50, "signal_hours": 4, "vol_ratio_min": 1.1, "vol_long": 60, "max_days": 6}
    assert check_lookahead(lambda: STRATEGIES.create("main_pullback_limit", params), md.bars(60)) == []
    s = STRATEGIES.create("main_pullback_limit", params)
    bars = md.bars(60)
    s.prepare(bars, np.random.default_rng(0))
    hours = (bars.t[s.decide] // 3_600_000_000_000) % 24
    assert set(hours.tolist()) <= {3, 7, 11, 15, 19, 23}        # 4 時間足の最後の 1 時間足が確定したとき
    assert not s.no_entry[~np.isin(((bars.t // 3_600_000_000_000) + 1) % 24, (20, 21, 22))].any()


def test_intraday_trades_are_short_and_fewer_with_the_volatility_gate():
    md = _hourly()
    base = {"signal_hours": 4, "limit_atr": 0.0, "max_days": 6}
    _, free = _run_limit(md, base)
    _, gated = _run_limit(md, {**base, "vol_ratio_min": 1.2, "vol_long": 100})
    assert 0 < len(gated.trades) < len(free.trades)
    held = np.array([(t.exit_t - t.entry_t) / 3.6e12 for t in free.trades])
    assert np.median(held) < 24


def test_rolling_version_decides_every_hour_without_lookahead_and_holds_one_position():
    from fxsim.backtest.lookahead import check_lookahead

    md = _hourly(years=1)
    params = {"slow": 50, "signal_hours": 4, "rolling": True, "vol_ratio_min": 1.1, "vol_long": 60, "max_days": 6}
    assert check_lookahead(lambda: STRATEGIES.create("main_pullback_limit", params), md.bars(60)) == []
    s, bt = _run_limit(_hourly(), {"signal_hours": 4, "rolling": True, "limit_atr": 0.0, "max_days": 6})
    assert len(s.ph) == 4 and s.decide.mean() > 0.95            # 4 つの位相のどれかが毎時確定する
    spans = sorted((t.entry_t, t.exit_t) for t in bt.trades)
    assert len(spans) > 10 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))   # 建玉は 1 つだけ
    _, fixed = _run_limit(_hourly(), {"signal_hours": 4, "limit_atr": 0.0, "max_days": 6})
    assert len(bt.trades) > len(fixed.trades)
