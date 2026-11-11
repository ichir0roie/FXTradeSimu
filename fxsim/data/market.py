"""市場モデルが出す価格列。基準解像度(既定 1 分)の mid とスプレッドを持つ。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bars import Bars, make_bars


@dataclass
class MarketData:
    times: np.ndarray          # int64 ns
    mid: np.ndarray            # float64
    spread: np.ndarray         # 価格単位(pip ではない)
    pip: float
    step_minutes: int
    events: list[dict] = field(default_factory=list)   # 悪意イベントの記録(idx, kind, ...)
    meta: dict = field(default_factory=dict)            # regime 列など(大きい配列を含みうる)
    model_info: dict = field(default_factory=dict)      # name / version / params
    _bars: dict = field(default_factory=dict, init=False, repr=False)   # 時間足のキャッシュ

    def __len__(self) -> int:
        return len(self.times)

    def bars(self, minutes: int) -> Bars:
        """minutes 分足。同じ足は 2 回目からキャッシュを返す。"""
        if minutes not in self._bars:
            b = make_bars(self.times, self.mid, self.spread, minutes, self.step_minutes)
            b.pair = self.meta.get("swap_pair")
            b.points_per_bar = int(self.meta.get("ohlc_points", 0))
            rd = self.meta.get("rate_diff_points")
            if rd is not None:
                b.rate_diff = np.asarray(rd, dtype=float)[b.start_idx]
            self._bars[minutes] = b
        return self._bars[minutes]
