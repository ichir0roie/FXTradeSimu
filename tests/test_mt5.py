"""MT5 のテスター(fxsim/backtest/mt5)の決まり。

約定の価格・発動の側・不正な要求・ストップアウト・ネットと両建て・期限・本命の戦略の共通の対策。
"""
import numpy as np
import pytest

from fxsim import catalog  # noqa: F401
from fxsim.backtest import Backtester, BrokerConfig, EngineConfig, compute_metrics
from fxsim.backtest.lookahead import check_lookahead
from fxsim.backtest.mt5 import Mt5Tester
from fxsim.backtest.mt5.spec import (
    DEAL_ENTRY_IN,
    DEAL_ENTRY_INOUT,
    ORDER_TIME_DAY,
    TRADE_RETCODE_INVALID_PRICE,
    TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_INVALID_VOLUME,
    TRADE_RETCODE_NO_MONEY,
)
from fxsim.calendar import trading_times
from fxsim.data.bars import make_bars
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from fxsim.strategies.guards import GuardConfig, near_rollover, size_lots

SPREAD = 0.004


def _bars(prices, tf: int = 60):
    p = np.asarray(prices, dtype=float)
    t = trading_times("2020-01-06", days=60, step_minutes=1)[: len(p)]
    return make_bars(t, p, np.full(len(p), SPREAD), tf)


def _broker(model: str = "every_tick", **mt5) -> BrokerConfig:
    return BrokerConfig.preset("fair", {"slippage_pips": 0.0, "swap_mode": "fixed",
                                        "mt5": {"tester_model": model, **mt5}})


class Script:
    """足の番号 -> ctx を受け取る関数。結果は self.results に残す。"""
    engine = "mt5"
    warmup = 0
    timeframe = 60
    magic = 1

    def __init__(self, actions):
        self.actions = actions
        self.results = {}

    def prepare(self, bars, rng):
        pass

    def on_bar(self, i, ctx):
        if i in self.actions:
            self.results[i] = self.actions[i](ctx)


def _run(prices, actions, broker=None, cfg=None):
    s = Script(actions)
    bt = Mt5Tester(_bars(prices), s, broker or _broker(), cfg or EngineConfig(), pip=0.01, seed=1).run()
    return bt, s


# ---- 1 ペアのエンジンとの一致 -----------------------------------------------------

class _ClassicFlip:
    warmup, timeframe = 0, 60

    def prepare(self, bars, rng):
        self.sig = np.sign(np.sin(np.arange(len(bars)) / 37.0))

    def on_bar(self, i, ctx):
        s = int(self.sig[i])
        if s != ctx.side:
            ctx.buy() if s > 0 else ctx.sell()


class _Mt5Flip(_ClassicFlip):
    engine = "mt5"

    def on_bar(self, i, ctx):
        s = int(self.sig[i])
        if s != int(np.sign(ctx.net_volume())):
            for p in ctx.positions():
                ctx.position_close(p.ticket)
            assert (ctx.buy(0.1) if s > 0 else ctx.sell(0.1)).ok


@pytest.mark.parametrize("preset", ["fair", "malicious"])
def test_market_orders_match_the_classic_engine(preset):
    """成行だけの戦略なら、約定(次の足の始値)・スプレッド・滑り・スワップが 1 ペアのエンジンと 1 円も違わない。"""
    md = MARKETS.create("malicious").generate(years=0.5, seed=3)
    bars = md.bars(60)
    br = BrokerConfig.preset(preset, {"last_look_prob": 0.0, "hunt_prob": 0.0})
    a = Backtester(bars, _ClassicFlip(), br, EngineConfig(), pip=md.pip, seed=1).run()
    b = Mt5Tester(bars, _Mt5Flip(), br, EngineConfig(), pip=md.pip, seed=1).run()
    assert len(a.trades) == len(b.trades) > 10
    assert b.balance == pytest.approx(a.balance, abs=1e-6)
    ma, mb = compute_metrics(a), compute_metrics(b)
    assert mb["costs"] == pytest.approx(ma["costs"], abs=1e-6)
    assert mb["mt5"]["total_trades"] == len(b.trades)


# ---- 待機注文と損切り・利確の約定 ---------------------------------------------------

def _dip_then_up(n_flat=120, low=99.5, n=300):
    p = np.full(n, 100.0)
    p[n_flat:n_flat + 30] = np.linspace(100.0, low, 30)
    p[n_flat + 30:] = np.linspace(low, 100.6, n - n_flat - 30)
    return p


def test_buy_limit_fills_at_the_limit_price_on_ask():
    bt, s = _run(_dip_then_up(), {0: lambda c: c.buy_limit(0.1, 99.8, tp=100.3)})
    assert s.results[0].retcode == 10008
    tr = bt.trades[0]
    assert tr.entry_price == pytest.approx(99.8)            # ask が 99.8 に届いた点で、指値の価格
    assert tr.reason == "tp" and tr.exit_price == pytest.approx(100.3)


def _gap_up(n=240, at=150, jump=1.0):
    p = np.full(n, 100.0)
    p[at:] = 100.0 + jump
    p += np.r_[0, np.cumsum(np.tile([0.001, -0.001], n // 2))[: n - 1]]     # ふつうの 1 歩(飛びの判定の基準)
    return p


def test_buy_stop_through_a_gap_fills_at_the_market_in_tick_modes():
    """逆指値は、窓で越えたら飛んだ先の気配で約定する(MT5 の every tick)。1 分足 OHLC と始値のみは注文の価格。"""
    act = {0: lambda c: c.buy_stop(0.1, 100.5)}
    bt, _ = _run(_gap_up(), act, _broker("every_tick"))
    assert bt.deals[0].price == pytest.approx(101.0 + SPREAD / 2, abs=0.002)
    bt, _ = _run(_gap_up(), act, _broker("m1_ohlc"))
    assert bt.deals[0].price == pytest.approx(100.5)


def test_sell_stop_loss_is_checked_on_ask():
    """売りの損切りは ask で判定する(mid が届かなくても、スプレッドの分だけ早く刺さる)。"""
    p = np.full(240, 100.0)
    p[100:] = 100.0 + 0.5 - SPREAD / 4          # mid は 100.5 に届かないが、ask は越える
    bt, _ = _run(p, {0: lambda c: c.sell(0.1, sl=100.5)})
    assert bt.trades[0].reason == "sl"


def test_invalid_requests_return_mt5_codes():
    big = 1e6 * 25 / 100.0 / 100_000 * 2         # 証拠金の倍の数量(ロット)
    acts = {
        0: lambda c: c.buy(0.1, sl=c.ask + 0.1),                 # 損切りが逆の側
        1: lambda c: c.buy(0.015),                               # 刻みに合わない
        2: lambda c: c.buy(round(big, 2)),                       # 証拠金が足りない
        3: lambda c: c.buy_limit(0.1, c.ask + 0.1),              # 指値が今の価格より上
    }
    bt, s = _run(np.full(400, 100.0), acts)
    assert [s.results[i].retcode for i in range(4)] == [TRADE_RETCODE_INVALID_STOPS, TRADE_RETCODE_INVALID_VOLUME,
                                                       TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_INVALID_PRICE]
    assert not bt.trades and bt.stats["no_money"] == 1


def test_stops_level_is_enforced():
    acts = {0: lambda c: c.buy(0.1, sl=c.bid - 0.009), 1: lambda c: c.buy(0.1, sl=c.bid - 0.011)}
    _, s = _run(np.full(200, 100.0), acts, _broker(stops_level=10))      # 10 ポイント = 0.010
    assert s.results[0].retcode == TRADE_RETCODE_INVALID_STOPS and s.results[1].ok


# ---- ストップアウト -----------------------------------------------------------------

def _crash(n=600):
    return np.r_[np.full(120, 100.0), np.linspace(100.0, 80.0, n - 120)]


def _hedged(c):
    c.buy(1.0)
    c.sell(0.5)


def test_stop_out_closes_the_largest_loss_first():
    bt, _ = _run(_crash(), {0: _hedged}, _broker(preset="jp_domestic", tester_model="every_tick"))
    lc = [t for t in bt.trades if t.reason == "losscut"]
    assert len(lc) == 1 and lc[0].side == 1                  # 負けている買いだけが閉じられ、売りは残る
    assert any(t.side == -1 and t.reason == "end" for t in bt.trades)
    assert bt.stats["stop_out"] == 1


def test_tester_style_stop_out_closes_everything():
    bt, _ = _run(_crash(), {0: _hedged}, _broker(preset="tester_ohlc"))
    assert sorted(t.side for t in bt.trades if t.reason == "losscut") == [-1, 1]


# ---- ネットと両建て -----------------------------------------------------------------

def test_netting_account_averages_and_reverses():
    p = np.full(300, 100.0)
    acts = {0: lambda c: c.buy(0.1), 1: lambda c: c.buy(0.1), 2: lambda c: c.sell(0.3)}
    bt, s = _run(p, acts, _broker(account_mode="netting"))
    assert all(s.results[i].ok for i in range(3))
    assert [d.entry for d in bt.deals[:4]] == [DEAL_ENTRY_IN, DEAL_ENTRY_IN, DEAL_ENTRY_INOUT, DEAL_ENTRY_IN]
    assert bt.trades[0].reason == "reverse" and bt.trades[0].units == pytest.approx(20_000)
    assert bt.trades[-1].side == -1 and bt.trades[-1].units == pytest.approx(10_000)


def test_close_by_nets_two_hedged_positions_at_the_opposite_open_price():
    p = np.r_[np.full(70, 100.0), np.full(200, 100.4)]

    def close_by(c):
        ps = {x.side: x.ticket for x in c.positions()}
        return c.position_close_by(ps[1], ps[-1])

    bt, s = _run(p, {0: lambda c: c.buy(0.2), 1: lambda c: c.sell(0.1), 2: close_by})
    assert s.results[2].ok
    by = [t for t in bt.trades if t.reason == "close_by"]
    assert len(by) == 2
    assert sum(t.pnl for t in by) == pytest.approx((100.4 - SPREAD / 2 - (100.0 + SPREAD / 2)) * 10_000, abs=1.0)
    assert [t.reason for t in bt.trades if t.reason != "close_by"] == ["end"]     # 残った買い 0.1


def test_day_orders_expire_at_the_end_of_the_trading_day():
    bt, s = _run(np.full(60 * 30, 100.0), {0: lambda c: c.buy_limit(0.1, 99.0, type_time=ORDER_TIME_DAY)})
    assert s.results[0].ok and not bt.orders and bt.stats["expired"] == 1


# ---- 本命の戦略と共通の対策 ---------------------------------------------------------

def test_main_strategy_has_no_lookahead():
    md = MARKETS.create("malicious").generate(years=2, seed=11)
    assert check_lookahead(lambda: STRATEGIES.create("main_trend_d1"), md.bars(1440)) == []


def test_size_lots_risks_a_fixed_fraction_of_equity():
    g = GuardConfig(risk_per_trade=0.01, max_leverage=50.0)
    _, s = _run(np.full(200, 100.0), {0: lambda c: size_lots(c, 0.5, g, 1)})
    # 100 万円 x 1% / 0.5 円 = 2 万通貨 = 0.2 ロット
    assert s.results[0] == pytest.approx(0.2)
    g = GuardConfig(risk_per_trade=0.01, max_leverage=1.0)        # 自分の上限: 100 万円分(ask 100.002 で 9,999 通貨)
    _, s = _run(np.full(200, 100.0), {0: lambda c: size_lots(c, 0.5, g, 1)})
    assert s.results[0] == pytest.approx(0.09)                    # 刻み 0.01 ロットに切り捨て


def test_near_rollover():
    day = np.datetime64("2020-01-07T00:00", "ns").astype(np.int64)
    hour = 3_600_000_000_000
    assert near_rollover(day + 22 * hour, 1.0) and near_rollover(day + 21 * hour + hour // 2, 1.0)
    assert not near_rollover(day + 12 * hour, 1.0)


def test_main_direction_null_keeps_timing_and_randomizes_sides():
    md = MARKETS.create("malicious").generate(years=6, seed=4)
    bars = md.bars(1440)
    guard = {"max_drawdown": 0.0, "max_trades_per_year": 0, "cost_multiple": 0.0}
    br = BrokerConfig.preset("fair")

    def entries(name, params):
        bt = Mt5Tester(bars, STRATEGIES.create(name, params), br, EngineConfig(), pip=md.pip, seed=2).run()
        return [(d.time, d.type) for d in bt.deals if d.entry == DEAL_ENTRY_IN]

    a = entries("main_trend_d1", {"guard": guard})
    b = entries("main_direction_null", {"inner": "main_trend_d1", "inner_params": {"guard": guard}})
    assert len(a) > 5 and a[0][0] == b[0][0]                 # 最初の新規は同じ時刻
    flipped = np.mean([x[1] != y[1] for x, y in zip(a, b, strict=False)])
    assert 0.15 < flipped < 0.85


# ---- 口座の床(資産の 9 割を守る) -----------------------------------------------------

def test_equity_floor_resets_each_month_and_halts_until_the_next():
    from fxsim.strategies.guards import EquityFloor
    day = 86_400 * 10**9
    jan = np.datetime64("2021-01-04", "ns").astype(np.int64)
    feb = np.datetime64("2021-02-01", "ns").astype(np.int64)
    f = EquityFloor(0.9)
    f.update(jan, 1_000_000)
    assert f.floor == pytest.approx(900_000)
    f.update(jan + 10 * day, 1_200_000)                     # 同じ月のうちは付け直さない
    assert f.floor == pytest.approx(900_000)
    assert f.room(950_000, 20_000) == pytest.approx(30_000)
    assert f.room(950_000, None) == 0.0                      # 逆指値の無い建玉があれば新規なし
    assert not f.check(900_001) and f.check(899_999) and f.halted
    assert not f.check(800_000)                              # 止めている間は二度手仕舞わない
    assert f.room(2_000_000, 0.0) == 0.0
    f.update(feb, 880_000)                                   # 月が変わったら付け直して再開
    assert not f.halted and f.floor == pytest.approx(792_000)


def test_floor_lots_keeps_the_worst_loss_within_the_room():
    from fxsim.strategies.guards import floor_lots, worst_loss
    # 1 ロット 10 万通貨、逆指値まで 1 円、窓で 1.5 倍 -> 1 ロットの最悪 15 万円
    assert floor_lots(0.5, 30_000, 150_000, 0.01) == pytest.approx(0.2)
    assert floor_lots(0.1, 30_000, 150_000, 0.01) == pytest.approx(0.1)
    assert floor_lots(0.5, float("inf"), 150_000, 0.01) == 0.5
    assert worst_loss(1, 10_000, 100.0, 99.0, 1.5) == pytest.approx(15_000)
    assert worst_loss(-1, 10_000, 100.0, 99.0, 1.5) == 0.0           # 利の乗った側の逆指値
    assert worst_loss(1, 10_000, 100.0, None, 1.5) is None


def _month_drops(bt, bars):
    """月ごとの (その月の最低の有効証拠金 / 月初の有効証拠金) の最小。"""
    months = bars.t.astype("datetime64[ns]").astype("datetime64[M]")
    eq = np.asarray(bt.equity_curve)
    worst = 1.0
    for m in np.unique(months):
        e = eq[months == m]
        worst = min(worst, float(e.min() / e[0]))
    return worst


def test_main_strategy_keeps_each_month_above_the_floor():
    md = MARKETS.create("malicious").generate(years=6, seed=4)
    bars = md.bars(1440)
    # 1 回 4% を賭ける荒い設定。床なしでは月に 1 割以上減る月がある
    guard = {"risk_per_trade": 0.04, "max_leverage": 50.0, "max_drawdown": 0.0, "max_trades_per_year": 0,
             "cost_multiple": 0.0}
    br = BrokerConfig.preset("fair")

    def run(protect):
        s = STRATEGIES.create("main_trend_d1", {"guard": {**guard, "protect": protect}})
        return Mt5Tester(bars, s, br, EngineConfig(), pip=md.pip, seed=2).run(), s

    bare, _ = run(0.0)
    held, s = run(0.9)
    assert _month_drops(bare, bars) < 0.9
    assert _month_drops(held, bars) >= 0.9 - 1e-9
    assert s.floor_cut + s.blocked["floor"] > 0
