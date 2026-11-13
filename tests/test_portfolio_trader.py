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
        self.extra: dict[str, dict] = {}            # 銘柄 -> symbol_info に足す値(スワップの点数・1 ティックの値)
        self.lose: str | None = None               # 次の order_send を切る: "before"(約定しない)/ "after"(約定した後)

    def _rates(self, name):
        return self.by_symbol[name][: self.visible]

    def symbol_info(self, name):
        d = self.digits[name]
        return SimpleNamespace(digits=d, point=10.0 ** -d, trade_contract_size=100_000, volume_min=0.01,
                               volume_max=100, volume_step=0.01, filling_mode=1, trade_stops_level=0,
                               **self.extra.get(name, {}))

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

    def order_send(self, req):
        lose, self.lose = self.lose, None
        if lose == "before":
            raise EOFError("lost before the order reached the server")
        r = super().order_send(req)
        if "position" not in req and req.get("action") != 6:
            self.positions[-1].comment = req.get("comment", "")     # 約定した時点でコメントが付く
        if lose == "after":
            raise EOFError("lost after the order was executed")
        return r


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
    kw = {"broker_swap_min_obs": 1, **kw}
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


def test_opens_are_tagged_with_the_sim_position_and_the_ticket_is_kept(tmp_path, monkeypatch):
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    tr.poll()
    forming = int(fake._rates("USDJPY")[-1]["time"]) - SERVER_OFFSET
    want = "fxsim:" + str(np.datetime64(forming, "s").astype("datetime64[D]").astype(object).strftime("%y%m%d"))
    assert {p.comment for p in fake.positions} == {want}
    st = TraderState.load(tr.state_path)
    assert st.pairs["USDJPY"]["tag"] == want and st.pairs["USDJPY"]["ticket"] > 0


@pytest.mark.parametrize("lose, n_positions_after_retry", [("after", 2), ("before", 2)])
def test_lost_orders_are_retried_but_never_doubled(tmp_path, monkeypatch, lose, n_positions_after_retry):
    """送信中に切れた注文: 約定した後に切れたなら出し直さない(同じ印の建玉がある)、約定前なら出し直す。"""
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    fake.lose = lose
    with pytest.raises(EOFError):
        tr.poll()                                   # 最初の銘柄の注文で切れる(常駐はつなぎ直して poll し直す)
    left = TraderState.load(tr.state_path).pending
    assert len(left) == 2                           # 切れた注文も含めて残っている
    tr.poll()
    assert len(fake.positions) == n_positions_after_retry
    assert sorted(p.symbol for p in fake.positions) == ["EURUSD", "USDJPY"]
    assert TraderState.load(tr.state_path).pending == []


def test_a_real_position_of_another_sim_position_is_closed_not_reopened(tmp_path, monkeypatch):
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    tr.poll()
    for p in fake.positions:
        p.comment = "fxsim:200101"                 # 前の(もう無い)シミュレーションの建玉に属する
    fake.visible += 1                               # 次の足
    tr.poll()
    assert fake.positions == []                     # 決済して、建て直さない(途中からは乗らない)


def test_broker_swap_decides_the_receiving_side(tmp_path, monkeypatch):
    """端末のスワップで、EURUSD は買いが受け取り・売りが払い(FRED の 2021 年とは逆)。下抜けでは売りを建てない。"""
    fake = _fake()
    fake.extra["EURUSD"] = {"swap_mode": 1, "swap_long": 5.0, "swap_short": -9.0}
    fake.extra["USDJPY"] = {"swap_mode": 1, "swap_long": 10.0, "swap_short": -25.0,
                            "trade_tick_value": 100.0, "trade_tick_size": 0.001}
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    tr.poll()
    assert {p.symbol for p in fake.positions} == {"USDJPY"}
    _, per, meta = tr.decide(int(fake._rates("USDJPY")[-1]["time"]))
    assert meta["broker_diff"]["EURUSD"] > 0 and meta["broker_diff"]["USDJPY"] > 0
    assert meta["k"]["USDJPY"] == pytest.approx(1.0)


def test_broker_swap_uses_the_median_of_daily_observations(tmp_path, monkeypatch):
    """OANDA の点数は次のロールオーバーで付く分(週末の分が付く日は 3 日分)。観測の中央値を 1 日分とする。"""
    fake = _fake()
    tr = _trader(fake, tmp_path, monkeypatch, broker_swap_min_obs=3)
    spec = SimpleNamespace(swap_mode=1, swap_long=10.0, swap_short=-25.0, point=0.001)
    assert tr.broker_diff("USDJPY", spec, 150.0, day=1) is None          # 観測が 3 回に満たない
    tr.broker_diff("USDJPY", spec, 150.0, day=2)
    spec.swap_long, spec.swap_short = 30.0, -75.0                          # 水曜は 3 日分
    d = tr.broker_diff("USDJPY", spec, 150.0, day=3)
    assert d == pytest.approx(10.0 * 0.001 * 365 / 150.0 + tr.sim_broker.swap_markup)
    assert tr.broker_diff("USDJPY", spec, 150.0, day=3) == pytest.approx(d)   # 同じ日の観測は 1 回だけ数える


def test_stops_converted_back_from_yen_are_not_moved_every_day():
    """円への換算(K)が日ごとに変わっても、銘柄の値段に戻した損切りが同じなら付け直さない(E027 の再生で毎日出ていた)。"""
    from fxsim.backtest.engine import Position
    from fxsim.trading.mt5 import LivePosition
    from fxsim.trading.portfolio import in_symbol_prices
    from fxsim.trading.signals import SimSnapshot
    from fxsim.trading.trader import plan

    def snap(k):
        pos = Position(side=-1, units=2000, entry_price=0.8 * k, entry_mid=0.8 * k, entry_idx=10,
                       sl=0.9 * k, tp=None)
        return in_symbol_prices(SimSnapshot(bar_t=1, position=pos, position_t=5), k)

    day1 = snap(187.3)
    assert day1.position.sl == pytest.approx(0.9) and day1.position.entry_price == pytest.approx(0.8)
    prev = TraderState(sim_position_t=5, sim_sl=day1.position.sl, sim_tp=None)
    real = [LivePosition(ticket=1, side=-1, volume=0.02, price_open=0.8, sl=0.9, tp=None, time=0)]
    for k in (186.1, 145.235, 182.889):            # 145.235 などは 0.9 * k / k が 0.9 にぴったり戻らない
        assert plan(snap(k), real, prev, lambda u: u / 100_000) == []
    moved = in_symbol_prices(SimSnapshot(bar_t=1, position=Position(-1, 2000, 0.8 * 190, 0.8 * 190, 10, 0.88 * 190,
                                                                     None), position_t=5), 190.0)
    assert [a.kind for a in plan(moved, real, prev, lambda u: u / 100_000)] == ["modify"]


def test_the_broker_carry_of_past_decision_bars_is_kept(tmp_path, monkeypatch):
    """端末のスワップで建てた判断が、次の日の計算でも残る(E027: 月次の金利の表に戻すと、FRED の向きでは建てない
    EURUSD の買いが次の日に消え、「シミュレーションは持っていない」で手仕舞っていた)。"""
    fake = _fake()
    eur = fake.by_symbol["EURUSD"]
    n = len(eur)
    for f in ("open", "high", "low", "close"):
        eur[f][n - 7:] += 0.06                     # 下抜けを上抜けに変える(2021 年の FRED では EURUSD は売りが受け取り)
    fake.extra["EURUSD"] = {"swap_mode": 1, "swap_long": 5.0, "swap_short": -9.0}   # 端末では買いが受け取り
    tr = _trader(fake, tmp_path, monkeypatch)
    tr.startup()
    tr.poll()
    assert {p.symbol: p.type for p in fake.positions}["EURUSD"] == 0
    for _ in range(2):                              # 次の日、その次の日
        fake.visible += 1
        tr.poll()
        assert {p.symbol: p.type for p in fake.positions}.get("EURUSD") == 0
    st = TraderState.load(tr.state_path)
    assert len(st.pairs["EURUSD"]["live_diff"]) == 4    # 判断の足 3 本 + 作りかけの足
