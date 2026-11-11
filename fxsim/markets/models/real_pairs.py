"""R013 real_pairs: 相場の DB の主要ペアを、日付をそろえて 1 つの口座で回す(複数ペアの戦略用)。

ペアごとに real_pair(R012: 円に換算、記録のスプレッド、MT5 のテスターと同じ足の中の順番、ペアの金利差)を作り、
全ペアにある足だけを残す(どれかのペアに無い日は落とす)。生成した MarketData は先頭のペアのもので、
全ペアは meta["legs"]。
複数ペアの戦略(Strategy.multi)は PortfolioBacktester で、証拠金と有効証拠金を共有して回る(basket_* と同じ)。
basket_* との違い: FRED の終値だけではなく MT5 の OHLC とスプレッド、対円以外のペア(円に換算)、足の中の経路(4 点)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ...data import MarketData
from ...registry import params_hash
from .. import MARKETS

DEFAULT_PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]


@dataclass
class RealPairsSource:
    timeframe: str = "D1"
    start: str | None = None
    end: str | None = None
    daily_fill_hour: int = 0


@dataclass
class RealPairsParams:
    pairs: list[str] = field(default_factory=lambda: list(DEFAULT_PAIRS))


class RealPairsModel:
    name = "real_pairs"
    version = "1"

    def __init__(self, params: dict | None = None):
        params = dict(params or {})
        self.src = RealPairsSource(**(params.pop("source", None) or {}))
        self.p = RealPairsParams(**params)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def source(self):
        return self.src

    def info(self) -> dict:
        params = {"source": asdict(self.src), **asdict(self.p)}
        return {"name": self.name, "version": self.version, "components": ["real_pair", "align"], "params": params,
                "params_hash": params_hash(params)}

    def generate(self, *, years: float | None = None, days: float | None = None, seed: int = 0) -> MarketData:
        legs: dict[str, MarketData] = {}
        for pair in self.p.pairs:
            src = {"path": f"db:{pair}/{self.src.timeframe}", "start": self.src.start, "end": self.src.end,
                   "daily_fill_hour": self.src.daily_fill_hour}
            legs[pair.lower()] = MARKETS.create("real_pair", {"source": src}).generate(years=years, days=days,
                                                                                      seed=seed)
        k = 4                                                  # real_pair は 1 本 = 4 点
        starts = {p: md.times[::k] for p, md in legs.items()}
        common = None
        for t in starts.values():
            common = t if common is None else np.intersect1d(common, t)
        if common is None or len(common) < 3:
            raise ValueError("real_pairs: too few bars shared by all pairs")
        info = self.info()
        out: dict[str, MarketData] = {}
        for p, md in legs.items():
            keep = np.repeat(np.isin(starts[p], common), k)[:len(md.times)]
            out[p] = MarketData(times=md.times[keep], mid=md.mid[keep], spread=md.spread[keep], pip=md.pip,
                                step_minutes=md.step_minutes, meta=dict(md.meta),
                                model_info={**info, "pair": p, "leg": md.model_info})
        first = out[self.p.pairs[0].lower()]
        first.meta["legs"] = out
        first.meta["real_source"] = f"db ({self.src.timeframe}, {len(out)} pairs)"
        return first


@MARKETS.register("real_pairs", version="1",
                  description="相場の DB の主要ペアを日付をそろえて 1 つの口座で回す(円に換算、記録のスプレッド、"
                              "ペアの金利差)",
                  tags=("real", "basket"))
def build_real_pairs(params: dict) -> RealPairsModel:
    return RealPairsModel(params)
