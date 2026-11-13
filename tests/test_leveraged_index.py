import numpy as np
import pytest

from fxsim.research.leveraged_index import NISA_LIFE, PLANS, TAX, IndexPlan, compare, shocks, simulate

ONE, TWO = PLANS


def _annuity(monthly: float, months: int, g: float) -> float:
    """毎月の頭に入れて、月末に exp(g) 倍になる積立の最終額。"""
    v = 0.0
    for _ in range(months):
        v = (v + monthly) * np.exp(g)
    return v


def test_zero_volatility_matches_the_closed_form():
    z = np.zeros((1, 120))
    mu, r = 0.06, 0.03
    one = simulate(ONE, z, mu, 0.0, r, 10_000)
    assert one["after_tax"][0] == pytest.approx(_annuity(10_000, 120, (mu - ONE.fee) / 12))
    # 2 倍型: 2 mu - r - fee。課税口座なので利益に 20.315%
    two = simulate(TWO, z, mu, 0.0, r, 10_000)
    pre = _annuity(10_000, 120, (2 * mu - r - TWO.fee) / 12)
    assert two["pre_tax"][0] == pytest.approx(pre)
    assert two["after_tax"][0] == pytest.approx(pre - (pre - 1_200_000) * TAX)


def test_volatility_drag_is_four_times_larger_at_2x():
    z = np.zeros((1, 12))
    sigma = 0.2
    plain = IndexPlan("x", "x", 1.0, 0.0, True)
    double = IndexPlan("y", "y", 2.0, 0.0, True)
    g1 = np.log(simulate(plain, z, 0.0, sigma, 0.0, 1.0)["pre_tax"][0] / _annuity(1.0, 12, 0.0))
    g2 = np.log(simulate(double, z, 0.0, sigma, 0.0, 1.0)["pre_tax"][0] / _annuity(1.0, 12, 0.0))
    assert g2 / g1 == pytest.approx(4.0, rel=0.05)


def test_no_tax_on_a_loss():
    z = np.full((1, 24), -3.0)
    out = simulate(TWO, z, 0.05, 0.2, 0.02, 10_000)
    assert out["after_tax"][0] < out["paid"]
    assert out["tax"][0] == 0.0


def test_nisa_lifetime_cap_overflows_to_a_taxable_account():
    months = 12 * 10
    monthly = 300_000                       # 年 360 万円・10 年で 3,600 万円。半分は枠を越える
    z = np.zeros((1, months))
    out = simulate(ONE, z, 0.05, 0.0, 0.0, monthly)
    assert out["tax"][0] > 0
    # 越えた分(後半 5 年)だけが課税される
    g = (0.05 - ONE.fee) / 12
    later = _annuity(monthly, 60, g)
    assert out["tax"][0] == pytest.approx((later - monthly * 60) * TAX)
    assert monthly * 60 == NISA_LIFE


def test_same_seed_same_result_and_common_random_numbers():
    a = compare(0.09, 0.18, 0.03, 5, 30_000, 200, seed=7)
    b = compare(0.09, 0.18, 0.03, 5, 30_000, 200, seed=7)
    assert np.array_equal(a["S051"]["after_tax"], b["S051"]["after_tax"])
    # 同じ乱数なので、基準価額の最大 DD は 2 倍型の方が大きいか同じ
    assert (a["S051"]["nav_mdd"] >= a["S050"]["nav_mdd"] - 1e-12).all()


def test_t_shocks_have_unit_variance():
    z = shocks(np.random.default_rng(0), 400, 500, df=5)
    assert z.var() == pytest.approx(1.0, rel=0.05)
    with pytest.raises(ValueError):
        shocks(np.random.default_rng(0), 1, 1, df=2)
