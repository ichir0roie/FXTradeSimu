"""fxsim trade(MT5 の常駐)の判断と注文。MT5 の代わりに同じ関数を持つ偽物(FakeMT5)を使う。"""
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig, EngineConfig
from fxsim.backtest.schedule import TradingSchedule
from fxsim.cases.model import BehaviorCase
from fxsim.strategies import STRATEGIES, Strategy
from fxsim.trading import TraderConfig, simulate
from fxsim.trading.feed import rates_to_bars
from fxsim.trading.mt5 import LivePosition, MT5Client, MT5Error
from fxsim.trading.signals import SimOrder, SimSnapshot
from fxsim.trading.trader import LiveTrader, TraderState, plan

H = 3600
SERVER_OFFSET = 2 * H                       # サーバー時刻 = UTC + 2 時間
T0 = 1_614_556_800 + SERVER_OFFSET          # 2021-03-01 00:00 UTC(月曜)をサーバー時刻で
RATE_DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
              ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]


def make_rates(closes, t0=T0, step=H, spread_points=4):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    r = np.zeros(len(c), dtype=RATE_DTYPE)
    r["time"] = t0 + np.arange(len(c)) * step
    r["open"], r["close"] = o, c
    r["high"], r["low"] = np.maximum(o, c) + 0.01, np.minimum(o, c) - 0.01
    r["tick_volume"], r["spread"] = 100, spread_points
    return r


class FakeMT5:
    """MetaTrader5 モジュールの偽物(使う関数だけ)。ヘッジ口座として建玉を持つ。"""
    TIMEFRAME_H1 = 16385
    ACCOUNT_TRADE_MODE_DEMO = 0

    def __init__(self, rates, demo=True):
        self.all_rates = rates
        self.visible = len(rates)          # いま見えている本数(最後の 1 本が作りかけ)
        self.demo = demo
        self.positions = []
        self.requests = []
        self.next_ticket = 1
        self.reject = None
        self.equity = 1e6

    # 接続
    def initialize(self, **kw):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(trade_allowed=True)

    def account_info(self):
        return SimpleNamespace(login=123, server="Demo", currency="JPY", balance=1e6, equity=self.equity, margin=0,
                               margin_free=1e6, leverage=25, trade_mode=0 if self.demo else 2, margin_mode=2,
                               company="x")

    # 銘柄・足
    def symbol_select(self, name, enable):
        return True

    def symbol_info(self, name):
        return SimpleNamespace(digits=3, point=0.001, trade_contract_size=100_000, volume_min=0.01, volume_max=100,
                               volume_step=0.01, filling_mode=1, trade_stops_level=0)

    def _now(self):
        return self.all_rates[: self.visible]

    def symbol_info_tick(self, name):
        last = self._now()[-1]
        bid = float(last["close"])
        return SimpleNamespace(time=int(time.time()) + SERVER_OFFSET, bid=bid, ask=bid + 0.004)

    def copy_rates_from_pos(self, name, tf, start, count):
        r = self._now()
        return r[max(0, len(r) - count):]

    def positions_get(self, symbol=None):
        return tuple(self.positions)

    def order_calc_profit(self, otype, symbol, volume, price_open, price_close):
        side = 1 if otype == 0 else -1
        return (price_close - price_open) * side * volume * 100_000

    def order_send(self, req):
        self.requests.append(req)
        if self.reject is not None:
            return SimpleNamespace(retcode=self.reject, comment="rejected", order=0, deal=0, price=0, volume=0)
        if req["action"] == 6:                                   # SLTP
            for p in self.positions:
                if p.ticket == req["position"]:
                    p.sl, p.tp = req["sl"], req["tp"]
        elif "position" in req:                                  # 決済
            for p in list(self.positions):
                if p.ticket == req["position"]:
                    p.volume = round(p.volume - req["volume"], 8)
                    if p.volume <= 0:
                        self.positions.remove(p)
        else:
            self.positions.append(SimpleNamespace(ticket=self.next_ticket, symbol=req["symbol"], type=req["type"],
                                                  volume=req["volume"],
                                                  price_open=req["price"], sl=req.get("sl", 0.0),
                                                  tp=req.get("tp", 0.0), time=int(self._now()[-1]["time"]),
                                                  magic=req["magic"]))
            self.next_ticket += 1
        return SimpleNamespace(retcode=10009, comment="done", order=1, deal=1, price=req.get("price", 0),
                               volume=req.get("volume", 0))


class Script(Strategy):
    """テスト用: 足の時刻(UTC ns の位置ではなく、起点からの本数)で決めた注文を出す。"""
    name = "script"

    @dataclass
    class Params:
        timeframe: int = 60

    def __init__(self, plan_):
        super().__init__()
        self.plan = plan_

    def on_bar(self, i, ctx):
        for act in self.plan.get(i, []):
            getattr(ctx, act[0])(**act[1])


def trader_for(fake, script, tmp_path, monkeypatch, **cfg_kw):
    monkeypatch.setattr("fxsim.trading.config.TRADING_DIR", tmp_path)
    cfg_kw = {"protect": 0.0, **cfg_kw}        # 口座の床は専用のテストで見る
    cfg = TraderConfig(name="t", behavior="BT", symbol="USDJPY", magic=77, history_bars=1000,
                       server_utc_offset_hours="auto", **cfg_kw)
    beh = BehaviorCase(id="BT", strategy="ma_cross", timeframe=60, engine={"units": 10000})
    return LiveTrader(cfg, MT5Client(fake), behavior=beh, strategy_factory=lambda: Script(script),
                      sleep=lambda s: None)


# ---- 足 -------------------------------------------------------------------

def test_rates_to_bars_converts_server_time_and_spread():
    r = make_rates([150.0, 150.1, 150.2], spread_points=4)
    b = rates_to_bars(r, 60, 0.001, SERVER_OFFSET)
    assert b.t[0] == np.datetime64("2021-03-01T00:00", "ns").astype(np.int64)
    np.testing.assert_allclose(b.ask_close - b.bid_close, 0.004)
    np.testing.assert_allclose(b.close, r["close"] + 0.002)
    assert b.base_mid is None


# ---- シミュレーションの写し ---------------------------------------------------

def _bars(n=30):
    return rates_to_bars(make_rates(150 + np.cumsum(np.full(n, 0.01))), 60, 0.001, SERVER_OFFSET)


def test_simulate_returns_orders_of_the_last_closed_bar():
    bars = _bars(30)
    snap = simulate(bars, Script({28: [("buy", {"sl": 0.5})]}), BrokerConfig.preset("fair"), EngineConfig(),
                    TradingSchedule(), 0.01)
    assert snap.bar_t == bars.t[28]
    assert snap.position is None
    assert [(o.kind, o.side, o.units, o.sl_dist) for o in snap.orders] == [("enter", 1, 10000.0, 0.5)]


def test_simulate_position_matches_the_backtest():
    """作りかけの足の手前までは、ふつうのバックテストと同じ建玉になる。"""
    bars = _bars(30)
    script = {5: [("buy", {"sl": 0.5})], 12: [("set_sl", {"price": 149.9})]}
    snap = simulate(bars, Script(script), BrokerConfig.preset("fair"), EngineConfig(), TradingSchedule(), 0.01)
    bt = Backtester(bars.slice(29), Script(script), BrokerConfig.preset("fair"), EngineConfig(), pip=0.01)
    bt.run()
    t = bt.trades[-1]            # 最後の足で "end" として閉じた建玉 = 写しの建玉
    assert snap.position.side == t.side and snap.position.entry_price == pytest.approx(t.entry_price)
    assert snap.position.sl == 149.9
    assert snap.position_t == bars.t[6]
    assert snap.orders == []


class Mt5Script(Script):
    """テスト用の本命の戦略(MT5 のテスターで回る)。("buy", {...}) / ("close_all", {}) を出す。"""
    engine = "mt5"

    def on_bar(self, i, ctx):
        for name, kw in self.plan.get(i, []):
            if name == "close_all":
                for p in ctx.positions():
                    ctx.position_close(p.ticket)
            else:
                getattr(ctx, name)(**kw)


def test_simulate_main_strategy_turns_mt5_fills_into_orders():
    bars = _bars(30)
    snap = simulate(bars, Mt5Script({28: [("buy", {"volume": 0.1, "sl": 149.0})]}), BrokerConfig.preset("fair"),
                    EngineConfig(), TradingSchedule(), 0.01)
    assert snap.bar_t == bars.t[28] and snap.position is None
    [o] = snap.orders
    assert (o.kind, o.side, o.units) == ("enter", 1, 10000.0)
    assert o.sl_dist == pytest.approx(bars.ask_open[29] - 149.0, abs=1e-3)


def test_simulate_main_strategy_reports_the_held_position_and_its_close():
    bars = _bars(30)
    held = simulate(bars, Mt5Script({5: [("buy", {"volume": 0.1})]}), BrokerConfig.preset("fair"),
                    EngineConfig(), TradingSchedule(), 0.01)
    assert held.position.side == 1 and held.position.units == 10000.0
    assert held.position_t == bars.t[6] and held.orders == []
    closed = simulate(bars, Mt5Script({5: [("buy", {"volume": 0.1})], 28: [("close_all", {})]}),
                      BrokerConfig.preset("fair"), EngineConfig(), TradingSchedule(), 0.01)
    assert closed.position.side == 1 and closed.position_t == bars.t[6]
    assert [o.kind for o in closed.orders] == ["close"]
    assert plan(closed, [_real(1)], TraderState(), lots)[0].kind == "close_all"


# ---- 突き合わせ ---------------------------------------------------------------

def _pos(side, sl=None, tp=None):
    from fxsim.backtest.engine import Position
    return Position(side, 10000, 150.0, 150.0, 3, sl, tp)


def _real(side, ticket=1):
    return LivePosition(ticket, side, 0.1, 150.0, None, None, 0)


def lots(u):
    return u / 100_000


def test_plan_enter_from_flat_and_skip_same_side():
    snap = SimSnapshot(0, None, None, [SimOrder("enter", 1, 10000, 0.5, None)])
    assert [(a.kind, a.side, a.lots) for a in plan(snap, [], TraderState(), lots)] == [("open", 1, 0.1)]
    snap = SimSnapshot(0, _pos(1), 0, [SimOrder("enter", 1, 10000)])
    assert plan(snap, [_real(1)], TraderState(), lots) == []


def test_plan_reverse_closes_then_opens():
    snap = SimSnapshot(0, _pos(1), 0, [SimOrder("enter", -1, 10000, 0.5)])
    assert [(a.kind, a.side) for a in plan(snap, [_real(1)], TraderState(), lots)] == [("close_all", 0), ("open", -1)]


def test_plan_follows_sim_exit_but_never_enters_on_its_own():
    # シミュレーションが損切りで閉じた → 実口座も閉じる
    assert [a.kind for a in plan(SimSnapshot(0, None, None, []), [_real(1)], TraderState(), lots)] == ["close_all"]
    # 実口座だけ閉じた(またはシミュレーションがすでに持っていた)→ 建て直さない
    assert plan(SimSnapshot(0, _pos(1), 0, []), [], TraderState(), lots) == []


def test_plan_moves_stops_only_for_the_same_sim_position():
    prev = TraderState(sim_position_t=7, sim_sl=149.5)
    snap = SimSnapshot(0, _pos(1, sl=149.8), 7, [])
    assert [(a.kind, a.sl) for a in plan(snap, [_real(1)], prev, lots)] == [("modify", 149.8)]
    snap = SimSnapshot(0, _pos(1, sl=149.8), 8, [])          # 別の建玉(建て直した)なら動かさない
    assert plan(snap, [_real(1)], prev, lots) == []


# ---- 注文まで ----------------------------------------------------------------

def test_trader_opens_once_per_bar_and_records_state(tmp_path, monkeypatch):
    rates = make_rates(150 + np.cumsum(np.full(40, 0.01)))
    fake = FakeMT5(rates)
    fake.visible = 31                       # 足 0..29 が確定、30 が作りかけ
    tr = trader_for(fake, {29: [("buy", {"sl": 0.5, "tp": 1.0})]}, tmp_path, monkeypatch)
    tr.startup()
    assert tr.poll() is True
    assert len(fake.positions) == 1
    p = fake.positions[0]
    assert p.type == 0 and p.volume == pytest.approx(0.1) and p.magic == 77
    assert p.sl == pytest.approx(p.price_open - 0.5) and p.tp == pytest.approx(p.price_open + 1.0)
    assert tr.poll() is False               # 同じ足は二度処理しない
    assert len(fake.requests) == 1
    saved = TraderState.load(tr.state_path)
    assert saved.utc_offset_sec == SERVER_OFFSET and saved.anchor == T0
    assert saved.last_bar_t == np.datetime64("2021-03-02T05:00", "ns").astype(np.int64)


def test_trader_closes_when_sim_closes_and_ignores_other_magic(tmp_path, monkeypatch):
    rates = make_rates(150 + np.cumsum(np.full(40, 0.01)))
    fake = FakeMT5(rates)
    fake.visible = 31
    tr = trader_for(fake, {28: [("buy", {})], 29: [("close", {})]}, tmp_path, monkeypatch)
    fake.positions.append(SimpleNamespace(ticket=50, type=0, volume=0.1, price_open=150, sl=0, tp=0, time=0, magic=77))
    fake.positions.append(SimpleNamespace(ticket=60, type=0, volume=0.3, price_open=150, sl=0, tp=0, time=0, magic=1))
    tr.poll()
    assert [p.ticket for p in fake.positions] == [60]       # 手動・ほかの EA の建玉には触らない


def test_dry_run_sends_nothing(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    tr = trader_for(fake, {29: [("buy", {})]}, tmp_path, monkeypatch, dry_run=True)
    assert tr.poll() is True
    assert fake.requests == [] and fake.positions == []
    assert tr.state_path.parent.name == "t.dry"


def test_max_lots_blocks_oversized_orders(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    tr = trader_for(fake, {29: [("buy", {"units": 500_000})]}, tmp_path, monkeypatch, max_lots=1.0)
    tr.poll()
    assert fake.requests == []


def test_refuses_real_account(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(np.full(10, 150.0)), demo=False)
    tr = trader_for(fake, {}, tmp_path, monkeypatch)
    with pytest.raises(MT5Error, match="real account but the config says account: demo"):
        tr.startup()


def test_trailing_stop_follows_sim_and_closes_when_crossed(tmp_path, monkeypatch):
    rates = make_rates(150 + np.cumsum(np.full(40, 0.01)))
    fake = FakeMT5(rates)
    fake.visible = 31
    script = {29: [("buy", {"sl": 0.5})], 31: [("set_sl", {"price": 150.1})], 32: [("set_sl", {"price": 151.0})]}
    tr = trader_for(fake, script, tmp_path, monkeypatch)
    tr.poll()
    sl0 = fake.positions[0].sl
    fake.visible = 32                        # 建てた直後の足: 実口座は約定値からの距離のまま
    tr.poll()
    assert fake.positions[0].sl == sl0
    fake.visible = 33
    tr.poll()
    assert fake.positions[0].sl == pytest.approx(150.1)
    fake.visible = 34                        # 151.0 は今の値段(約 150.34)より上 = もう越えている
    tr.poll()
    assert fake.positions == []


def test_retries_requotes(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    tr = trader_for(fake, {29: [("buy", {})]}, tmp_path, monkeypatch)
    fake.reject = 10004
    tr.poll()
    assert len(fake.requests) == 4           # 1 回 + やり直し 3 回
    assert fake.positions == []


def test_example_configs_load():
    from fxsim.cases.loader import load_cases
    from fxsim.paths import CONFIGS_DIR
    from fxsim.trading import load_trader_config
    behaviors = load_cases().behaviors
    paths = sorted((CONFIGS_DIR / "trading").glob("*.yaml"))
    assert paths
    for p in paths:
        cfg = load_trader_config(p)
        assert cfg.behavior in behaviors and not cfg.allow_real
        multi = STRATEGIES.create(behaviors[cfg.behavior].strategy, behaviors[cfg.behavior].strategy_params()).multi
        assert multi == bool(cfg.symbols)          # 複数ペアの戦略は symbols を書く(PortfolioTrader)


def test_remote_results_become_plain_objects():
    """Wine 側の名前付きタプル(手元に MetaTrader5 が無いと pickle を読めない)は SimpleNamespace にして受け取る。"""
    from fxsim.trading.mt5 import _from_plain
    pos = {"__nt__": True, "ticket": 5, "type": 0, "magic": 77}
    out = _from_plain((pos, pos))
    assert out[0].ticket == 5 and out[1].magic == 77 and not hasattr(out[0], "__nt__")
    assert _from_plain((0, "ok")) == (0, "ok")


def test_server_offset_ignores_unset_tick_time():
    """ログイン直後は気配の時刻が 0 で、差が「-2 年」と出る(実際に OANDA のデモで起きた)。"""
    fake = FakeMT5(make_rates(np.full(5, 150.0)))
    client = MT5Client(fake)
    assert client.server_offset("USDJPY") == SERVER_OFFSET
    fake.symbol_info_tick = lambda name: SimpleNamespace(time=0, bid=150.0, ask=150.004)
    assert client.server_offset("USDJPY") is None


def test_order_send_falls_back_to_keyword_arguments():
    """端末のビルド 6314 からは order_send(dict) が (-2, 'Unnamed arguments not allowed') で None を返す
    (OANDA のデモで実際に起きた)。キーワード引数で渡し直して通す。"""
    fake = FakeMT5(make_rates(np.full(5, 150.0)))
    real = fake.order_send
    err = [(1, "Success")]

    def strict(req=None, **kw):
        if req is not None:
            err[0] = (-2, "Unnamed arguments not allowed")
            return None
        err[0] = (1, "Success")
        return real(kw)
    fake.order_send = strict
    fake.last_error = lambda: err[0]
    r = MT5Client(fake).market("USDJPY", -1, 0.01, sl=150.05, tp=149.95, magic=7, deviation=20, comment="x")
    assert r.ok and len(fake.positions) == 1 and fake.positions[0].magic == 7


def test_rates_retry_until_history_arrives():
    fake = FakeMT5(make_rates(np.full(5, 150.0)))
    real = fake.copy_rates_from_pos
    calls = []

    def flaky(*a):
        calls.append(a)
        return None if len(calls) < 3 else real(*a)
    fake.copy_rates_from_pos = flaky
    client = MT5Client(fake, sleep=lambda s: None)
    assert len(client.rates("USDJPY", 60, 5)) == 5 and len(calls) == 3


def test_real_account_needs_allow_real():
    with pytest.raises(ValueError, match="allow_real"):
        TraderConfig(name="r", behavior="B001", symbol="USDJPY", magic=1, account="real")
    assert TraderConfig(name="r", behavior="B001", symbol="USDJPY", magic=1, account="real", allow_real=True)


def test_demo_config_refuses_terminal_logged_in_to_another_account(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(np.full(10, 150.0)))
    tr = trader_for(fake, {}, tmp_path, monkeypatch)
    tr.expected_login = 999                       # FakeMT5 の口座は 123
    with pytest.raises(MT5Error, match="not 999"):
        tr.startup()


def test_load_credentials_reads_account_file(tmp_path):
    from fxsim.trading.config import load_credentials
    assert load_credentials("demo", tmp_path / ".env.demo") is None
    f = tmp_path / ".env.demo"
    f.write_text("# demo\nMT5_LOGIN=12345678\nMT5_PASSWORD='a*b!c%d'\nMT5_SERVER='OANDA-Japan MT5 Demo'\n",
                 encoding="utf-8")
    c = load_credentials("demo", f)
    assert (c.login, c.password, c.server) == (12345678, "a*b!c%d", "OANDA-Japan MT5 Demo")
    assert "a*b" not in repr(c)                    # パスワードはログに出さない
    f.write_text("MT5_LOGIN=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MT5_PASSWORD"):
        load_credentials("demo", f)


# ---- 口座の床 -------------------------------------------------------------------

def _manual(fake, side=1, volume=0.3, sl_below=1.0, magic=1, ticket=90):
    bid = float(fake._now()[-1]["close"])
    sl = bid - side * sl_below if sl_below is not None else 0.0
    fake.positions.append(SimpleNamespace(ticket=ticket, symbol="USDJPY", type=0 if side > 0 else 1, volume=volume,
                                          price_open=bid, sl=sl, tp=0.0, time=0, magic=magic))


def test_floor_cuts_new_entries_so_the_worst_case_stays_above_it(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    _manual(fake)               # 手動の買い 0.3 ロット、逆指値まで 1 円 -> 最悪 0.3 x 10 万 x 1 x 1.5 = 4.5 万円
    tr = trader_for(fake, {29: [("buy", {"units": 100_000, "sl": 0.5})]}, tmp_path, monkeypatch, protect=0.9)
    tr.poll()
    # 床 90 万円、余裕 100 - 4.5 - 90 = 5.5 万円。1 ロットの最悪 0.5 x 10 万 x 1.5 = 7.5 万円 -> 0.73 ロット
    mine = [p for p in fake.positions if p.magic == 77]
    assert len(mine) == 1 and mine[0].volume == pytest.approx(0.73)
    saved = json.loads((tmp_path / "account-123.json").read_text())
    assert saved["floor"] == pytest.approx(900_000) and not saved["halted"]


def test_position_without_a_stop_blocks_new_entries(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    _manual(fake, sl_below=None)
    tr = trader_for(fake, {29: [("buy", {"sl": 0.5})]}, tmp_path, monkeypatch, protect=0.9)
    tr.poll()
    assert [p.magic for p in fake.positions] == [1]


def test_floor_breach_closes_own_positions_and_halts_for_the_month(tmp_path, monkeypatch):
    fake = FakeMT5(make_rates(150 + np.cumsum(np.full(40, 0.01))))
    fake.visible = 31
    _manual(fake)
    tr = trader_for(fake, {29: [("buy", {"sl": 0.5})], 31: [("close", {})], 32: [("buy", {"sl": 0.5})]},
                    tmp_path, monkeypatch, protect=0.9)
    tr.poll()
    assert any(p.magic == 77 for p in fake.positions)
    fake.equity = 890_000                      # 月初 100 万円の 9 割を割った
    tr._floor_checked = 0.0
    tr.poll()                                  # 足は進んでいないが、床は見る
    assert [p.magic for p in fake.positions] == [1]          # 自分の建玉だけ決済
    fake.equity = 950_000
    fake.visible = 34                          # 足 32 の買いも、その月は出さない
    tr.poll()
    assert [p.magic for p in fake.positions] == [1]
    assert json.loads((tmp_path / "account-123.json").read_text())["halted"]


def test_poll_interval_follows_the_timeframe():
    cfg = TraderConfig(name="t", behavior="BT", symbol="USDJPY", magic=1)
    assert cfg.poll_interval(1440) == 60 and cfg.poll_interval(60) == 30 and cfg.poll_interval(5) == 5
    assert TraderConfig(name="t", behavior="BT", symbol="USDJPY", magic=1, poll_seconds=7).poll_interval(1440) == 7


@pytest.mark.parametrize("side, opens", [("buy", True), ("sell", False)])
def test_swap_side_only_behaviors_still_open_on_the_receiving_side(tmp_path, monkeypatch, side, opens):
    """常駐のシミュレーションのブローカーはスワップを固定の 0 にしている。swap_side_only の動作ケースでは、
    動作ケースのブローカーの金利差で向きを決める(0 のままだと全部の新規が止まっていた)。
    2021-03 の USD − JPY は +0.23%"""
    monkeypatch.setattr("fxsim.trading.config.TRADING_DIR", tmp_path)
    rates = make_rates(150 + np.cumsum(np.full(40, 0.01)))
    fake = FakeMT5(rates)
    fake.visible = 31
    cfg = TraderConfig(name="t", behavior="BT", symbol="USDJPY", magic=77, history_bars=1000, protect=0.0)
    beh = BehaviorCase(id="BT", strategy="ma_cross", timeframe=60, engine={"units": 10000, "swap_side_only": True},
                       broker={"preset": "fair", "overrides": {"swap_markup": 0.0}})
    tr = LiveTrader(cfg, MT5Client(fake), behavior=beh,
                    strategy_factory=lambda: Script({29: [(side, {"sl": 0.5})]}), sleep=lambda s: None)
    tr.startup()
    assert tr.sim_broker.swap_mode == "rates" and tr.sim_broker.slippage_pips == 0.0
    tr.poll()
    assert (len(fake.positions) == 1) == opens
