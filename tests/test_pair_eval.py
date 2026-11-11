"""実データ(相場の DB)の 1 ペアの再生(real_pair)と、ペア x 期間の評価(fxsim/pair_eval.py、E016)。"""
import numpy as np
import pytest

from fxsim import paths
from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.cases.runner import to_experiment
from fxsim.data import load_ohlc, load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS, RANGES, pair_eval_case
from fxsim.storage import prices
from fxsim.strategies import STRATEGIES

SRC = "Test-Demo"
T0 = 1_600_000_000 // 86_400 * 86_400 + 86_400 * 3      # 木曜 0 時 UTC あたり


def _write(con, symbol, tf, closes, spread_pts=5, start=T0, opens=None):
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    o = c if opens is None else np.asarray(opens, dtype=float)
    digits = 3 if symbol.endswith("JPY") else 5
    prices.upsert_symbol(con, SRC, symbol, digits, 10.0 ** -digits, "UTC+2")
    t = start + np.arange(n, dtype=np.int64) * tf * 60
    sp = np.full(n, spread_pts, dtype=np.int64) if np.isscalar(spread_pts) else np.asarray(spread_pts, np.int64)
    rows = {"open": o, "high": np.maximum(o, c) + 0.0002 * c, "low": np.minimum(o, c) - 0.0002 * c, "close": c,
            "tick_volume": np.full(n, 10), "spread": sp, "real_volume": np.zeros(n, np.int64),
            "server_time": t + 7200, "time": t}
    prices.write_rates(con, SRC, symbol, tf, rows, "UTC+2")


@pytest.fixture
def market_db(tmp_path, monkeypatch):
    path = tmp_path / "market.sqlite"
    monkeypatch.setattr(paths, "MARKET_DB", path)
    load_ohlc.cache_clear()
    con = prices.open_market_db(path)
    _write(con, "USDJPY", 1440, [150.0] * 30, start=T0 - 20 * 86_400)
    n = 24 * 20
    rng = np.random.default_rng(1)
    eur = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    sp = np.full(n, 6)
    sp[100] = 60                                            # ロールオーバーのような 1 本だけ広い足
    _write(con, "EURUSD", 60, eur, spread_pts=sp, opens=np.r_[eur[0], eur[:-1]])
    con.commit()
    con.close()
    yield path
    load_ohlc.cache_clear()


def _md(path="db:EURUSD/H1"):
    return MARKETS.create("real_pair", {"source": {"path": path}}).generate(years=1.0, seed=1)


def test_real_pair_converts_the_quote_currency_to_yen(market_db):
    md = _md()
    s = load_ohlc("db:EURUSD/H1")
    k = md.meta["quote_to_jpy"]                     # USDJPY の mid(DB の bid + 半スプレッド)
    assert k == pytest.approx(150.0, rel=1e-4)
    assert md.pip == pytest.approx(0.0001 * k)
    assert md.meta["swap_pair"] == "eurusd"
    b = md.bars(60)
    assert b.pair == "eurusd"
    assert b.close[:5] == pytest.approx(s.close[:5] * k)
    # 4 時間足にまとめても同じ換算(値動きの比率は変わらない)
    b4 = md.bars(240)
    assert b4.pair == "eurusd" and b4.close[-1] == pytest.approx(s.close[-1] * k)


def test_real_pair_orders_the_bar_like_the_mt5_tester(market_db):
    md = _md()
    s = load_ohlc("db:EURUSD/H1")
    k = md.meta["quote_to_jpy"]
    for i in range(5):
        pts = md.mid[4 * i:4 * i + 4] / k
        up = s.close[i] >= s.open[i]
        assert pts[0] == pytest.approx(s.open[i])
        assert pts[1] == pytest.approx(s.low[i] if up else s.high[i])
        assert pts[2] == pytest.approx(s.high[i] if up else s.low[i])
        assert pts[3] == pytest.approx(s.close[i])


def test_recorded_spread_widens_the_bar_edges_next_to_a_wide_bar(market_db):
    md = _md()
    sp = md.spread.reshape(-1, 4) / md.pip          # 足 x 点(始値・安高・安高・終値)の pip
    assert sp[100] == pytest.approx([6.0] * 4)
    assert sp[99] == pytest.approx([0.6, 0.6, 0.6, 6.0])       # 広い足の手前: 終値の点だけ
    assert sp[101] == pytest.approx([6.0, 0.6, 0.6, 0.6])      # 広い足の後: 始値の点だけ
    assert sp[50] == pytest.approx([0.6] * 4)
    b = md.bars(60)
    assert b.ask_close[99] - b.bid_close[99] == pytest.approx(6.0 * md.pip)
    assert b.ask_close[98] - b.bid_close[98] == pytest.approx(0.6 * md.pip)


def test_backtests_on_real_pair_pay_the_pair_rate_differential(market_db):
    md = _md()
    strat = STRATEGIES.create("buy_and_hold", {"timeframe": 60})
    bt = build_backtester(md, strat, BrokerConfig(swap_mode="rates"), EngineConfig(), seed=1)
    want = load_rates(MULTI_RATES).pair_differential("eurusd", md.bars(60).t)
    assert np.allclose(bt.rate_diff, want)


def test_carry_trade_reads_the_pair_rates_on_real_pair(market_db):
    md = _md()
    b = md.bars(60)
    s = STRATEGIES.create("carry_trade", {"timeframe": 60, "min_diff": 0.0})
    s.prepare(b, np.random.default_rng(1))
    diff = load_rates(MULTI_RATES).pair_differential("eurusd", b.t)
    assert np.array_equal(s.want, np.where(diff >= 0, 1, -1))


def test_pair_eval_case_crosses_pairs_and_windows_and_runs_the_null():
    book = load_cases()
    for bid, rng_key in [("B001", "H1"), ("B004", "D1"), ("B031", "M5")]:
        t = pair_eval_case(book.behaviors[bid])
        r = RANGES[rng_key]
        assert len(t.patterns) == len(PAIRS) * len(r.windows)
        assert {p.params["source"]["path"] for p in t.patterns} == {f"db:{p}/{r.data_tf}" for p in PAIRS}
        t.validate(book.behaviors)
        cfg = to_experiment(t, book.behaviors[bid])
        assert [s.label for s in cfg.strategies] == [bid, "NULL"]
        # 期間はパターンごと(テストケースの期間で上書きしない)
        starts = {m.params["source"]["start"] for m in cfg.markets}
        assert starts == {w[0] for w in r.windows}
    basket = pair_eval_case(book.behaviors["B011"])
    assert {p.market for p in basket.patterns} == {"basket_real"}
    assert len(basket.patterns) == len(RANGES["basket"].windows)


class _BuyOnce:
    """足 0 の終値で、損切り sl_dist の買いを 1 回だけ出す(足 1 の始値で約定)。"""
    multi = False
    warmup = 0

    def __init__(self, sl_dist, timeframe=60):
        self.sl_dist, self.timeframe = sl_dist, timeframe

    def prepare(self, bars, rng):
        pass

    def on_bar(self, i, ctx):
        if i == 0:
            ctx.buy(sl=self.sl_dist, units=10_000)


def _usdjpy_h1(con, opens, highs, lows, closes):
    prices.upsert_symbol(con, SRC, "USDJPY", 3, 0.001, "UTC+2")
    n = len(closes)
    t = T0 + np.arange(n, dtype=np.int64) * 3600
    rows = {"open": np.asarray(opens, float), "high": np.asarray(highs, float), "low": np.asarray(lows, float),
            "close": np.asarray(closes, float), "tick_volume": np.full(n, 10), "spread": np.zeros(n, np.int64),
            "real_volume": np.zeros(n, np.int64), "server_time": t + 7200, "time": t}
    prices.write_rates(con, SRC, "USDJPY", 60, rows, "UTC+2")
    con.commit()


@pytest.mark.parametrize("bar2_open, want_fill", [(149.95, 149.80), (149.70, 149.70)])
def test_real_ohlc_stops_fill_at_the_level_inside_a_bar_and_at_the_open_across_a_gap(tmp_path, monkeypatch,
                                                                                       bar2_open, want_fill):
    """足の中で損切りに届いたら損切りの価格(足の中は連続)。足と足の間の窓で越えたら次の足の始値。"""
    path = tmp_path / "market.sqlite"
    monkeypatch.setattr(paths, "MARKET_DB", path)
    load_ohlc.cache_clear()
    con = prices.open_market_db(path)
    #          足 0    足 1(買いが約定)  足 2(安値 149.50 まで)  足 3
    _usdjpy_h1(con, [150.0, 150.0, bar2_open, 149.6], [150.1, 150.1, bar2_open + 0.05, 149.8],
               [149.9, 149.95, 149.50, 149.5], [150.0, 150.0, 149.6, 149.7])
    con.close()
    md = MARKETS.create("real_pair", {"source": {"path": "db:USDJPY/H1"}, "spread": {"floor_pips": 0.0}}) \
        .generate(years=1.0, seed=1)
    load_ohlc.cache_clear()
    broker = BrokerConfig(slippage_pips=0.0, touch="bridge")
    bt = build_backtester(md, _BuyOnce(sl_dist=0.20), broker, EngineConfig(), seed=1)
    bt.run()
    tr = bt.trades[0]
    assert tr.reason == "sl"
    assert tr.entry_price == pytest.approx(150.0)
    assert tr.exit_price == pytest.approx(want_fill)


def test_oanda_swap_markup_is_smaller_on_the_receiving_side():
    b = BrokerConfig.preset("oanda")
    assert b.swap_annual(1, 0.03) == pytest.approx(0.03 - 0.0044)       # 受け取る側
    assert b.swap_annual(-1, 0.03) == pytest.approx(-0.03 - 0.011)      # 払う側
    m = BrokerConfig.preset("malicious")
    assert m.swap_annual(1, 0.03) == pytest.approx(0.02) and m.swap_annual(-1, 0.03) == pytest.approx(-0.04)


def test_pair_eval_can_swap_the_broker_and_keep_the_mt5_account():
    book = load_cases()
    t = pair_eval_case(book.behaviors["B045"], broker="oanda", daily_fill_hour=2)
    assert t.id == "P-B045-oanda-h2"
    assert t.broker == {"preset": "oanda", "label": "oanda", "overrides": {"mt5": {"preset": "jp_domestic"}}}
    assert {p.params["source"]["daily_fill_hour"] for p in t.patterns} == {2}
    cfg = to_experiment(t, book.behaviors["B045"])
    assert cfg.brokers[0].preset == "oanda"
    h1 = pair_eval_case(book.behaviors["B001"], broker="oanda", daily_fill_hour=2)
    assert h1.id == "P-B001-oanda" and "daily_fill_hour" not in h1.patterns[0].params["source"]


def test_daily_open_pays_the_rollover_spread_unless_the_fill_is_delayed(tmp_path, monkeypatch):
    """日足の始値の点は、1 分足のサーバーの 0 時台(ロールオーバー)の平均スプレッド。daily_fill_hour で遅らせられる。"""
    path = tmp_path / "market.sqlite"
    monkeypatch.setattr(paths, "MARKET_DB", path)
    load_ohlc.cache_clear()
    con = prices.open_market_db(path)
    # 1 分足 3 日分。サーバー時刻 = UTC + 2 なので、サーバーの 0 時台は UTC 22 時台。そこだけ 5 pip、ほかは 0.5 pip
    n = 3 * 1440
    t = T0 + np.arange(n, dtype=np.int64) * 60
    sp = np.where(((t // 3600) % 24) == 22, 50, 5)
    _write(con, "USDJPY", 1, np.full(n, 150.0), spread_pts=sp, start=T0)
    _write(con, "USDJPY", 1440, [150.0] * 30, start=T0 - 20 * 86_400)
    con.commit()
    con.close()
    for hour, want in [(0, 5.0), (2, 0.5)]:
        md = MARKETS.create("real_pair", {"source": {"path": "db:USDJPY/D1", "daily_fill_hour": hour}}) \
            .generate(years=1.0, seed=1)
        pts = md.spread.reshape(-1, 4) / md.pip
        assert np.median(pts[:, 0]) == pytest.approx(want)               # 始値の点
        assert np.median(pts[:, 1]) == pytest.approx((5.0 + 23 * 0.5) / 24, abs=0.05)   # 足の中は 1 日の平均
    load_ohlc.cache_clear()
