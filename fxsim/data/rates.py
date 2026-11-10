"""金利の表(スワップの計算とキャリーの判断に使う)。

CSV(列 date と通貨ごとの列、年率 %)の各行は「その日から次の行の前日まで有効」。表より前の日付は最初の行、
後の日付は最後の行の値を使う。
- `data/real/policy_rates_usd_jpy.csv`(列 usd, jpy)は手入力の概算(README を参照)。1 ペアのバックテストの既定
- `data/real/short_rates_monthly_fred.csv`(列 usd, jpy, eur, gbp, aud, nzd, cad, chf)は FRED の短期金利の月次。
  複数ペアのバックテストの既定(`fxsim/data/fred.py` で作る)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..paths import resolve

DEFAULT_RATES = "data/real/policy_rates_usd_jpy.csv"
MULTI_RATES = "data/real/short_rates_monthly_fred.csv"


@dataclass(frozen=True)
class RateTable:
    t: np.ndarray        # 有効になる時刻(int64 ns)
    base: np.ndarray     # 基軸通貨(USD)の年率(小数)
    quote: np.ndarray    # 決済通貨(JPY)の年率(小数)
    cols: dict = field(default_factory=dict, compare=False)   # 通貨(小文字)-> 年率(小数)の配列

    def at(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """各時刻で有効な (基軸の金利, 決済の金利)。"""
        i = self._index(t_ns)
        return self.base[i], self.quote[i]

    def _index(self, t_ns: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.t, t_ns, side="right") - 1, 0, len(self.t) - 1)

    def differential(self, t_ns: np.ndarray) -> np.ndarray:
        """金利差(基軸 − 決済)。買いはこれを受け取り、売りは払う(ブローカーの上乗せは別)。"""
        b, q = self.at(t_ns)
        return b - q

    def rate(self, ccy: str, t_ns: np.ndarray) -> np.ndarray:
        """通貨 ccy の各時刻の年率(小数)。"""
        c = ccy.lower()
        if c not in self.cols:
            raise KeyError(f"rate table has no column {c!r} (has {sorted(self.cols)})")
        return self.cols[c][self._index(t_ns)]

    def pair_differential(self, pair: str, t_ns: np.ndarray) -> np.ndarray:
        """通貨ペア(例 'eurjpy')の金利差(基軸 − 決済)。"""
        p = pair.lower()
        return self.rate(p[:3], t_ns) - self.rate(p[3:6], t_ns)


@lru_cache(maxsize=4)
def load_rates(path: str = DEFAULT_RATES) -> RateTable:
    df = pd.read_csv(resolve(path)).sort_values("date")
    t = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    cols = {c.lower(): df[c].to_numpy(float) / 100 for c in df.columns if c != "date"}
    return RateTable(t, cols.get("usd", np.zeros(len(t))), cols.get("jpy", np.zeros(len(t))), cols)
