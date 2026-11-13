"""S036 xs_gbdt: 勾配ブースティング(LightGBM)で、翌 horizon 本のペアの相対の伸びを予測し、その順位で売買する。

特徴量(足 i の終値までで計算できるもの、ペアごと)
    5 / 21 / 63 / 126 / 252 本の log リターン、63 本の実現ボラ、200 本移動平均からの乖離、金利差(年率)、
    21 本・63 本のリターンと金利差の「全ペア平均との差」
目的変数
    horizon 本先までの log リターン + 金利差 x horizon / 260(スワップの分)を、全ペアの平均との差にしたもの
学習(walk-forward)
    特徴量がそろってから min_train 本たったところで最初に学習し、以後 retrain 本ごとに学習し直す。
    学習に使うのは「答え(horizon 本先の値)が学習する足の終値までに出ている」標本だけ。7 ペアをまとめて 1 つのモデル。
    最初の学習までは持たない。乱数の種・スレッド 1 本で、同じシードなら同じ結果になる。
LightGBM が無い環境では prepare で ImportError になる(`pip install lightgbm`)。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..basket import MULTI_RATES, CrossSectional, lagged_log_return, rate_differentials, rolling_vol
from ..indicators import sma

FEATURES = ("r5", "r21", "r63", "r126", "r252", "vol63", "ma200", "carry", "r21_x", "r63_x", "carry_x")


def features(c: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """(足 x ペア x 特徴量)。"""
    r = {k: lagged_log_return(c, k) for k in (5, 21, 63, 126, 252)}
    vol = rolling_vol(c, 63)
    ma = np.column_stack([sma(c[:, j], 200) for j in range(c.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.log(c / ma)

    def rel(x):
        with warnings.catch_warnings():             # 全ペアが nan の足(先頭)の平均は nan のまま
            warnings.simplefilter("ignore", RuntimeWarning)
            return x - np.nanmean(x, axis=1, keepdims=True)

    cols = [r[5], r[21], r[63], r[126], r[252], vol, dev, diff, rel(r[21]), rel(r[63]), rel(diff)]
    return np.stack(cols, axis=2)


@STRATEGIES.register("xs_gbdt", version="1",
                     description="LightGBM で翌月のペアの相対の伸びを予測し、上位を買い下位を売る"
                                 "(1 年ごとに学習し直す)")
class XsGbdt(CrossSectional):
    name = "xs_gbdt"

    @dataclass
    class Params(CrossSectional.Params):
        horizon: int = 21
        min_train: int = 756        # 最初の学習までに要る本数(特徴量がそろってから)
        retrain: int = 252
        num_rounds: int = 200
        learning_rate: float = 0.03
        num_leaves: int = 7
        min_data_in_leaf: int = 100
        seed: int = 7
        rates_path: str = MULTI_RATES

    def _params(self) -> dict:
        p = self.p
        return {"objective": "regression", "learning_rate": p.learning_rate, "num_leaves": p.num_leaves,
                "min_data_in_leaf": p.min_data_in_leaf, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 1, "lambda_l2": 1.0, "seed": p.seed, "deterministic": True, "num_threads": 1,
                "force_row_wise": True, "verbose": -1}

    def score(self, bars_by_pair, c):
        import lightgbm as lgb

        t = bars_by_pair[self.pairs[0]].t
        diff = rate_differentials(self.pairs, t, self.p.rates_path)
        x = features(c, diff)
        n, k, _ = x.shape
        h = self.p.horizon
        y = np.full((n, k), np.nan)
        y[:n - h] = np.log(c[h:] / c[:n - h]) + diff[:n - h] * h / 260
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = y - np.nanmean(y, axis=1, keepdims=True)
        ok = np.isfinite(x).all(axis=2)
        ready = np.flatnonzero(ok.all(axis=1))
        score = np.full((n, k), np.nan)
        if len(ready) == 0:
            return score
        start = int(ready[0]) + self.p.min_train
        self.models = []
        for r in range(start, n, max(1, self.p.retrain)):
            rows = np.arange(0, r - h + 1)                     # 答え(rows + h)が足 r の終値までに出ている標本
            m = ok[rows] & np.isfinite(y[rows])
            xi, yi = x[rows][m], y[rows][m]
            if len(yi) < 10 * self.p.min_data_in_leaf:
                continue
            model = lgb.train(self._params(), lgb.Dataset(xi, yi, feature_name=list(FEATURES)),
                              num_boost_round=self.p.num_rounds)
            self.models.append((r, model))
            end = min(n, r + self.p.retrain)
            seg = x[r:end].reshape(-1, len(FEATURES))
            pred = model.predict(seg).reshape(end - r, k)
            score[r:end] = np.where(ok[r:end], pred, np.nan)
        return score
