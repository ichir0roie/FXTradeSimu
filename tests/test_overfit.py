import numpy as np
import pytest

from fxsim.overfit import _norm_ppf, deflated_sharpe, expected_max_sharpe, pbo


def test_norm_ppf():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.01) == pytest.approx(-2.326348, abs=1e-5)


def test_pbo_is_half_for_noise_and_low_for_a_real_edge():
    # 1 回の乱数の PBO は大きくばらつく(全期間で偶然強かった戦略は、どちらの半分でも上に来やすい)ので平均で見る
    vals = [pbo(np.random.default_rng(s).standard_normal((40, 30)), 10)["pbo"] for s in range(30)]
    assert 0.35 < np.mean(vals) < 0.65
    noise = np.random.default_rng(0).standard_normal((40, 30))
    edge = noise.copy()
    edge[:, 5] += 1.0
    assert pbo(edge, 10)["pbo"] < 0.05


def test_deflated_sharpe_penalizes_many_trials():
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0) > 0
    sr = 1.0 / np.sqrt(260)        # 年率 1.0
    few = deflated_sharpe(sr, 2600, 2)["dsr"]
    many = deflated_sharpe(sr, 2600, 1000)["dsr"]
    assert few > 0.95 > many
    assert deflated_sharpe(0.0, 2600, 1)["psr"] == pytest.approx(0.5)
