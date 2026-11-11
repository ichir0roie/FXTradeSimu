"""複数ペアの常駐(fxsim/trading/portfolio.py の PortfolioTrader)。MT5 の代わりに 2 銘柄の偽物を使う。"""
from types import SimpleNamespace

import numpy as np
import pytest
from test_trading import SERVER_OFFSET, T0, FakeMT5, make_rates

from fxsim.cases.model import BehaviorCase
from fxsim.strategies import STRATEGIES
from fxsim.trading import TraderConfig
from fxsim.trading.mt5 import MT5Client
from fxsim.trading.portfolio import PortfolioTrader
from fxsim.trading.trader import TraderState

DAY = 86_400


class FakeMulti(FakeMT5):
    """銘柄ごとに足と桁数を持つ偽物。サーバー時刻は server_now で決める。"""
    TIMEFRAME_D1 = 16408

    def __init__(self, rates_by_symbol, digits):
        super().__init__(next(iter(rates_by_symbol.values())))
        self.by_symbol = rates_by_symbol
        self.digits = digits
        self.server_now = None

    def _rates(self, name):
        return self.by_symbol[name][: self.visible]

    def symbol_info(self, name):
        d = self.digits[name]
        return SimpleNamespace(digits=d, point=10.0 ** -d, trade_contract_size=100_000, volume_min=0.01,
                               volume_max=100, volume_step=0.01, filling_mode=1, trade_stops_level=0)

    def symbol_info_tick(self, name):
        last = self._rates(name)[-1]
        bid = float(last["close"])
        t = self.server_now if self.server_now is not None else int(last["time"]) + 60
        return SimpleNamespace(time=t, bid=bid, ask=bid + 10.0 ** -self.digits[name] * 4)

    def copy_rates_from_pos(self, name, tf, start, count):
        r = self._rates(name)
        return r[max(0, len(r) - count):]

    def positions_get(self, symbol=None):
        return tuple(p for p in self.positions if symbol is None or p.symbol == symbol)


def _fake(n=40):
    """もみ合いの後、最後の確定足(n − 7)で USDJPY が上に、EURUSD が下に抜ける。足 n − 6 が作りかけ。"""
    rng = np.random.default_rng(1)
    jpy = 100 + rng.normal(0, 0.05, n)
    eur = 1.20 + rng.normal(0, 0.0005, n)
    jpy[n - 7:] += 3.0
    eur[n - 7:] -= 0.03
    up = make_rates(jpy, t0=T0, step=DAY)
    down = make_rates(eur, t0=T0, step=DAY)
    # 平日だけ。MT5 の日足はサーバーの 0 時に始まる(T0 はサーバー時刻の 2 時なので 2 時間戻す。T0 は月曜)
    days = [T0 - SERVER_OFFSET + k * DAY for k in range(3 * n) if (k % 7) < 5][:n]
    up["time"] = down["time"] = days
    down["high"], down["low"] = np.maximum(down["open"], down["close"]) + 0.0002, \
        np.minimum(down["open"], down["close"]) - 0.0002
    fake = FakeMulti({"USDJPY": up, "EURUSD": down}, {"USDJPY": 3, "EURUSD": 5})
    fake.visible = n - 5
    return fake


def _trader(fake, tmp_path, monkeypatch, **kw):
    monkeypatch.setattr("fxsim.trading.config.TRADING_DIR", tmp_path)
    cfg = TraderConfig(name="p", behavior="BP", symbol="USDJPY", symbols=["USDJPY", "EURUSD"], magic=88,
                       history_bars=1000, protect=0.0, server_utc_offset_hours=SERVER_OFFSET / 3600, **kw)
    beh = BehaviorCase(id="BP", strategy="main_carry_breakout", timeframe=1440, broker="fair",
                       engine={"initial_balance": 1e6})
    params = {"n_in": 5, "n_out": 3, "min_carry": -1.0, "c2r_min": 0.0, "vol_n": 5, "atr_n": 5, "protect": 0.0,
              "max_spread_mult": 0.0, "units": 10_000, "margin_cap": 1.0}
    return PortfolioTrader(cfg, MT5Client(fake), behavior=beh,
                           strategy_factory=lambda: STRATEGIES.create("main_carry_breakout", params),
                           sleep=lambda s: None)


def test_portfolio_trader_opens_the_receiving_side_of_each_pair_with_stops_in_symbol_prices(tmp_path, monkeypatch):
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    assert tr.poll() is True
    by = {p.symbol: p for p in fake.positions}
    assert set(by) == {"USDJPY", "EURUSD"}
    # 2021-03 の金利差: USD − JPY はプラス(買い)、EUR − USD はマイナス(売り)。上げ・下げのブレイクアウトに合う
    assert by["USDJPY"].type == 0 and by["EURUSD"].type == 1
    assert by["USDJPY"].volume == pytest.approx(0.1) and by["EURUSD"].volume == pytest.approx(0.1)
    # 急変用の逆指値は ATR x 9。円に換算した幅を銘柄の値段に戻しているので、EURUSD の幅はドルの値段の大きさ
    d_jpy = by["USDJPY"].price_open - by["USDJPY"].sl
    d_eur = by["EURUSD"].sl - by["EURUSD"].price_open
    assert 1.0 < d_jpy < 20.0 and 0.005 < d_eur < 0.2
    assert tr.poll() is False                       # 同じ足は二度処理しない


def test_portfolio_trader_waits_for_the_fill_hour(tmp_path, monkeypatch):
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch, fill_server_hour=2)
    tr.startup()
    forming = int(fake._rates("USDJPY")[-1]["time"])
    fake.server_now = forming + 60                  # サーバーの 0 時 1 分
    assert tr.poll() is True
    assert fake.positions == []
    saved = TraderState.load(tr.state_path)
    assert {p["symbol"] for p in saved.pending} == {"USDJPY", "EURUSD"}
    fake.server_now = forming + 2 * 3600 + 5        # 2 時を過ぎた
    assert tr.poll() is True                        # 待っていた注文を出す
    assert {p.symbol for p in fake.positions} == {"USDJPY", "EURUSD"}
    assert TraderState.load(tr.state_path).pending == []
    n = len(fake.requests)
    tr.poll()
    assert len(fake.requests) == n                  # 二度は出さない


def test_portfolio_trader_waits_until_every_symbol_has_the_forming_bar(tmp_path, monkeypatch):
    fake = _fake()
    fake.by_symbol["EURUSD"] = fake.by_symbol["EURUSD"][:-1]      # EURUSD の最新の足はまだ無い
    fake.visible = 40
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    assert tr.poll() is False and fake.requests == []
    assert TraderState.load(tr.state_path).last_forming is None    # 処理済みにしない(次の回にやり直す)
