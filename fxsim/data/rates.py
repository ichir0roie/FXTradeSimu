"""政策金利の表(スワップの計算に使う)。

CSV(列 date, usd, jpy、年率 %)の各行は「その日から次の行の前日まで有効」。表より前の日付は最初の行、
後の日付は最後の行の値を使う。同梱の `data/real/policy_rates_usd_jpy.csv` は手入力の概算(README を参照)。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..paths import resolve

DEFAULT_RATES = "data/real/policy_rates_usd_jpy.csv"


@dataclass(frozen=True)
class RateTable:
    t: np.ndarray        # 有効になる時刻(int64 ns)
    base: np.ndarray     # 基軸通貨(USD)の年率(小数)
    quote: np.ndarray    # 決済通貨(JPY)の年率(小数)

    def at(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """各時刻で有効な (基軸の金利, 決済の金利)。"""
        i = np.clip(np.searchsorted(self.t, t_ns, side="right") - 1, 0, len(self.t) - 1)
        return self.base[i], self.quote[i]

    def differential(self, t_ns: np.ndarray) -> np.ndarray:
        """金利差(基軸 − 決済)。買いはこれを受け取り、売りは払う(ブローカーの上乗せは別)。"""
        b, q = self.at(t_ns)
        return b - q


@lru_cache(maxsize=4)
def load_rates(path: str = DEFAULT_RATES) -> RateTable:
    df = pd.read_csv(resolve(path)).sort_values("date")
    t = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    return RateTable(t, df["usd"].to_numpy(float) / 100, df["jpy"].to_numpy(float) / 100)
