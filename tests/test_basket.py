from dataclasses import dataclass

import numpy as np
import pytest

from fxsim.backtest import Backtester, BrokerConfig, EngineConfig, compute_metrics
from fxsim.backtest.portfolio import PortfolioBacktester, check_lookahead_portfolio
from fxsim.data import load_rates
from fxsim.data.rates import DEFAULT_RATES, MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES, Strategy
from fxsim.strategies.basket import BasketStrategy, lagged_log_return, rank_sides


def _legs(model="basket_bootstrap", years=6, seed=3):
    return MARKETS.create(model).generate(years=years, seed=seed).meta["legs"]


def test_basket_models_keep_cross_correlation_and_shared_blocks():
    real = {p: m.bars(1440).close for p, m in _legs("basket_real").items()}
    boot = {p: m.bars(1440).close for p, m in _legs("basket_bootstrap").items()}
    r_real = np.array([np.diff(np.log(c)) for c in real.values()])
    r_boot = np.array([np.diff(np.log(c)) for c in boot.values()])
    assert np.allclose(np.corrcoef(r_real), np.corrcoef(r_boot), atol=0.08)
    # 並べ替えは全ペアで同じ日付の並び: 各日のリターンの組が、実データのどこかの日の組と一致する
    rows = {tuple(np.round(x, 12)) for x in r_real.T}
    assert np.mean([tuple(np.round(x, 12)) in rows for x in r_boot.T]) > 0.99


def test_basket_placebo_flips_all_pairs_together():
    real = np.array([np.diff(np.log(m.bars(1440).close)) for m in _legs("basket_real").values()])
    pl = np.array([np.diff(np.log(m.bars(1440).close)) for m in _legs("basket_placebo").values()])
    ratio = pl / np.where(real == 0, np.nan, real)
    s = np.nanmedian(ratio, axis=0)
    assert set(np.round(s).astype(int)) <= {-1, 1}
    assert np.nanmax(np.abs(ratio - s)) < 1e-6


def test_basket_real_is_deterministic_and_aligned():
    a, b = _legs("basket_real", seed=1), _legs("basket_real", seed=2)
    assert list(a) == ["usdjpy", "eurjpy", "gbpjpy", "audjpy", "nzdjpy", "cadjpy", "chfjpy"]
    t0 = a["usdjpy"].bars(1440).t
    for p in a:
        assert np.array_equal(a[p].mid, b[p].mid)
        assert np.array_equal(a[p].bars(1440).t, t0)


def test_rate_table_pair_differential():
    rt = load_rates(MULTI_RATES)
    t = np.array([np.datetime64("2008-10-15", "ns").astype(np.int64)])
    assert rt.pair_differential("audjpy", t)[0] == pytest.approx(rt.rate("aud", t)[0] - rt.rate("jpy", t)[0])
    old = load_rates(DEFAULT_RATES)
    assert np.allclose(old.pair_differential("usdjpy", t), old.differential(t))


class _Single(Strategy):
    name = "single_script"

    @dataclass
    class Params:
        timeframe: int = 1440

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def on_bar(self, i, ctx):
        s = self.plan.get(i)
        if s is None:
            return
        if s == 0:
            ctx.close()
        elif s > 0:
            ctx.buy()
        else:
            ctx.sell()


class _Multi(BasketStrategy):
    name = "multi_script"

    def __init__(self, plan, pair="usdjpy", units=None):
        super().__init__()
        self.plan, self.pair, self.units = plan, pair, units

    def prepare(self, bars_by_pair, rng):
        pass

    def on_bar(self, i, ctx):
        s = self.plan.get(i)
        if s is not None:
            ctx.target(self.pair, s, self.units)


def test_single_leg_portfolio_matches_the_single_pair_engine():
    bars = _legs("basket_real", years=4)["usdjpy"].bars(1440)
    rng = np.random.default_rng(0)
    plan = {int(i): int(rng.choice([-1, 0, 1])) for i in np.sort(rng.choice(len(bars) - 2, 60, replace=False))}
    broker = BrokerConfig.preset("malicious", {"hunt_prob": 0.0})
    cfg = EngineConfig()
    single = Backtester(bars, _Single(plan), broker, cfg, seed=5).run()
    multi = PortfolioBacktester({"usdjpy": bars}, _Multi(plan), broker, cfg, seed=5, rates_path=DEFAULT_RATES).run()
    assert len(single.trades) == len(multi.trades) > 10
    assert multi.equity_curve[-1] == pytest.approx(single.equity_curve[-1], abs=1e-6)
    assert np.allclose(multi.equity_curve, single.equity_curve)
    ms, mm = compute_metrics(single), compute_metrics(multi)
    for k in ("final_equity", "n_trades", "swap_total", "gross_pips_per_year"):
        assert mm[k] == pytest.approx(ms[k])
    assert all(t.pair == "usdjpy" for t in multi.trades)


def test_portfolio_shares_margin_across_pairs():
    legs = _legs("basket_real", years=2)
    bars = {p: legs[p].bars(1440) for p in ("usdjpy", "eurjpy")}

    class Both(BasketStrategy):
        name = "both"

        def prepare(self, b, rng):
            pass

        def on_bar(self, i, ctx):
            if i == 5:
                ctx.target("usdjpy", 1, 120_000)    # 約 1,900 万円(証拠金 25 倍の上限は約 2,500 万円)
            if i == 7:
                ctx.target("eurjpy", 1, 150_000)

    pb = PortfolioBacktester(bars, Both(), BrokerConfig.preset("fair"), EngineConfig(max_units=1e9)).run()
    units = {t.pair: t.units for t in pb.trades}
    assert units["usdjpy"] == 120_000
    assert 1_000 <= units["eurjpy"] < 60_000          # 残りの証拠金の分だけ


def test_account_losscut_closes_every_pair():
    legs = _legs("basket_real", years=3)
    bars = {p: legs[p].bars(1440) for p in ("usdjpy", "eurjpy", "gbpjpy")}

    class Max(BasketStrategy):
        name = "max"

        def prepare(self, b, rng):
            pass

        def on_bar(self, i, ctx):
            if i == 1:
                for p in ctx.pairs:
                    ctx.target(p, -1, 1e9)

    pb = PortfolioBacktester(bars, Max(), BrokerConfig.preset("fair"),
                             EngineConfig(max_units=1e9, initial_balance=1e6)).run()
    reasons = [t.reason for t in pb.trades]
    assert "losscut" in reasons
    assert len({t.exit_t for t in pb.trades if t.reason == "losscut"}) == 1


def test_rank_sides_and_lagged_returns():
    assert list(rank_sides(np.array([0.3, np.nan, 0.1, 0.5, -0.2]), 1, 1)) == [0, 0, 0, 1, -1]
    assert list(rank_sides(np.array([0.3, np.nan, np.nan]), 1, 1)) == [0, 0, 0]
    c = np.exp(np.arange(10.0))[:, None]
    r = lagged_log_return(c, 3, 1)
    assert np.isnan(r[2, 0]) and r[3, 0] == pytest.approx(2.0) and r[9, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("name", ["xs_carry", "xs_momentum", "xs_value", "basket_trend", "basket_direction_null"])
def test_basket_strategies_do_not_look_ahead(name):
    legs = _legs("basket_bootstrap", years=8)
    assert check_lookahead_portfolio(lambda: STRATEGIES.create(name), {p: m.bars(1440) for p, m in legs.items()}) == []


def test_basket_null_keeps_the_timing_of_the_inner_strategy():
    legs = {p: m.bars(1440) for p, m in _legs("basket_bootstrap", years=8).items()}
    a = PortfolioBacktester(legs, STRATEGIES.create("xs_momentum"), seed=2).run()
    b = PortfolioBacktester(legs, STRATEGIES.create("basket_direction_null", {"inner": "xs_momentum"}), seed=2).run()
    # 向きを決め直して今と同じ向きが出たら注文しない(1 ペアの direction_null と同じ)ので、判断は中の戦略の部分集合
    da = {(i, p) for i, p, *_ in a.decisions}
    db = {(i, p) for i, p, *_ in b.decisions}
    assert db <= da and len(db) > 0.5 * len(da)
    assert [d[3] for d in a.decisions] != [d[3] for d in b.decisions]


def test_single_pair_strategies_run_on_the_first_leg_of_a_basket_market():
    from fxsim.experiments import ExperimentConfig, run_experiment
    from fxsim.storage import connect

    cfg = ExperimentConfig.from_dict({"name": "t", "years": 3, "seeds": [1], "workers": 1,
                                      "markets": ["basket_bootstrap", "random_walk"],
                                      "strategies": [{"name": "carry_trade"}, {"name": "xs_carry"}]})
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        db = f"{d}/t.sqlite"
        run_id = run_experiment(cfg, db, log=lambda *a: None)
        rows = connect(db).execute("SELECT p.market, j.strategy, j.status FROM jobs j JOIN paths p ON p.id = j.path_id"
                                   " WHERE j.run_id=?", (run_id,)).fetchall()
    st = {(m, s): x for m, s, x in rows}
    assert st[("basket_bootstrap", "carry_trade")] == "done"
    assert st[("basket_bootstrap", "xs_carry")] == "done"
    assert st[("random_walk", "xs_carry")] == "skipped"


def test_reduce_keeps_the_account_consistent():
    """一部決済・積み増しを混ぜても、最終の残高 = 初期 + 取引の損益の合計(スワップ・手数料込み)。"""
    md = MARKETS.create("real_bridge").generate(years=2, seed=4)
    for broker in ("fair", "ecn", "malicious"):
        bt = Backtester(md.bars(60), STRATEGIES.create("repeat_grid", {"step_pct": 0.3}),
                        BrokerConfig.preset(broker, {"zero_cut": True}), seed=4).run()
        reasons = {t.reason for t in bt.trades}
        assert "reduce" in reasons
        assert bt.balance == pytest.approx(bt.cfg.initial_balance + sum(t.pnl for t in bt.trades), abs=1e-3)
        c = compute_metrics(bt)["costs"]
        assert c["gross_pnl"] - c["total"] == pytest.approx(sum(t.pnl for t in bt.trades), rel=1e-6, abs=1.0)


def test_repeat_grid_holds_more_levels_as_price_falls():
    md = MARKETS.create("real_bridge").generate(years=2, seed=4)
    bars = md.bars(60)
    bt = Backtester(bars, STRATEGIES.create("repeat_grid", {"step_pct": 0.3}), BrokerConfig.preset("fair"),
                    seed=4).run()
    assert all(t.side == 1 for t in bt.trades)
    assert max(t.units for t in bt.trades) >= 2000


def test_verification_of_a_multi_pair_behavior_uses_basket_patterns_and_null():
    from fxsim.cases import verification_case
    from fxsim.cases.loader import load_cases
    from fxsim.cases.runner import to_experiment
    from fxsim.verification import BASKET_PATTERNS

    t = verification_case("B011", years=30, seeds=[1, 2], block_days=250)
    assert [p.market for p in t.patterns] == BASKET_PATTERNS
    assert t.period["start"] == "1999-01-04"
    cfg = to_experiment(t, load_cases().behaviors["B011"])
    assert [s.name for s in cfg.strategies] == ["xs_carry", "basket_direction_null"]
    boot = next(m for m in cfg.markets if m.model == "basket_bootstrap")
    assert boot.params["source"]["block_days"] == 250
