"""R013 real_pairs: 相場の DB の主要ペアを、日付をそろえて 1 つの口座で回す(複数ペアの戦略用)。

ペアごとに real_pair(R012: 円に換算、記録のスプレッド、MT5 のテスターと同じ足の中の順番、ペアの金利差)を作り、
全ペアにある足だけを残す(どれかのペアに無い日は落とす)。生成した MarketData は先頭のペアのもので、
全ペアは meta["legs"]。
複数ペアの戦略(Strategy.multi)は PortfolioBacktester で、証拠金と有効証拠金を共有して回る(basket_* と同じ)。
basket_* との違い: FRED の終値だけではなく MT5 の OHLC とスプレッド、対円以外のペア(円に換算)、足の中の経路(4 点)。

R014 real_pairs_bootstrap: 同じ 10 ペアの「別の歴史」。全ペアで同じ日付のブロック(block_days 本)を、元の期間(pool)から
復元抽出でつなぎ、値動き(足の 4 点の形ごと)と、その日付の金利差をまとめて運ぶ。ペアどうしの相関、ボラの塊、
金利差と値動きのつながり(高金利通貨の値下がり・円高のときの金利差)が保たれ、日付の並びだけが変わる。
スプレッドは運ばず、置く先の期間の記録を使う(業者の費用は今の口座のもの。1990 年代の広いスプレッドを持ち込まない)。
これまでの合成の相場は金利差を本当の日付のまま当てたので、キャリーが有利に見えた(ROADMAP の「合成の相場のスワップ」)。
金利差は Bars.rate_diff でエンジンと戦略に渡る。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from ...data import MarketData
from ...data.rates import MULTI_RATES, load_rates
from ...registry import params_hash
from .. import MARKETS

DEFAULT_PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]


@dataclass
class RealPairsSource:
    timeframe: str = "D1"
    start: str | None = None
    end: str | None = None
    daily_fill_hour: int = 0
    # 並べ替え(real_pairs_bootstrap)だけ: ブロックの長さ(本)と、ブロックを取ってくる元の期間(省くと DB の全部)
    block_days: int = 250
    pool_start: str | None = None
    pool_end: str | None = None


@dataclass
class RealPairsParams:
    pairs: list[str] = field(default_factory=lambda: list(DEFAULT_PAIRS))


class RealPairsModel:
    name = "real_pairs"
    version = "1"
    mode = "replay"

    def __init__(self, params: dict | None = None):
        params = dict(params or {})
        self.src = RealPairsSource(**(params.pop("source", None) or {}))
        self.p = RealPairsParams(**params)

    @property
    def deterministic(self) -> bool:
        return self.mode == "replay"

    @property
    def source(self):
        return self.src

    def info(self) -> dict:
        params = {"source": asdict(self.src), **asdict(self.p)}
        return {"name": self.name, "version": self.version, "components": ["real_pair", "align"], "params": params,
                "params_hash": params_hash(params)}

    def generate(self, *, years: float | None = None, days: float | None = None, seed: int = 0) -> MarketData:
        out = _aligned(tuple(self.p.pairs), self.src.timeframe, self.src.start, self.src.end,
                       int(self.src.daily_fill_hour), years, days)
        if self.mode == "bootstrap":
            pool = _aligned(tuple(self.p.pairs), self.src.timeframe, self.src.pool_start, self.src.pool_end,
                            int(self.src.daily_fill_hour), None, None)
            out = _bootstrap(out, pool, int(self.src.block_days), np.random.default_rng([seed, 1313]))
        info = self.info()
        legs = {}
        for p, md in out.items():
            legs[p] = MarketData(times=md.times, mid=md.mid, spread=md.spread, pip=md.pip,
                                 step_minutes=md.step_minutes, meta=dict(md.meta),
                                 model_info={**info, "pair": p, "leg": md.model_info})
        first = legs[self.p.pairs[0].lower()]
        first.meta["legs"] = legs
        first.meta["real_source"] = f"db ({self.src.timeframe}, {len(legs)} pairs, {self.mode})"
        return first


class RealPairsBootstrap(RealPairsModel):
    name = "real_pairs_bootstrap"
    version = "1"
    mode = "bootstrap"


@lru_cache(maxsize=8)
def _aligned(pairs: tuple, timeframe: str, start, end, fill_hour: int, years, days) -> dict[str, MarketData]:
    """全ペアにある足だけを残した real_pair のペア -> MarketData(読み取り専用に扱う)。"""
    legs: dict[str, MarketData] = {}
    for pair in pairs:
        src = {"path": f"db:{pair}/{timeframe}", "start": start, "end": end, "daily_fill_hour": fill_hour}
        legs[pair.lower()] = MARKETS.create("real_pair", {"source": src}).generate(years=years, days=days, seed=0)
    k = 4                                                  # real_pair は 1 本 = 4 点
    starts = {p: md.times[::k] for p, md in legs.items()}
    common = None
    for t in starts.values():
        common = t if common is None else np.intersect1d(common, t)
    if common is None or len(common) < 3:
        raise ValueError("real_pairs: too few bars shared by all pairs")
    out: dict[str, MarketData] = {}
    for p, md in legs.items():
        keep = np.repeat(np.isin(starts[p], common), k)[:len(md.times)]
        out[p] = MarketData(times=md.times[keep], mid=md.mid[keep], spread=md.spread[keep], pip=md.pip,
                            step_minutes=md.step_minutes, meta=dict(md.meta), model_info=md.model_info)
    return out


def _bootstrap(target: dict[str, MarketData], pool: dict[str, MarketData], block: int,
               rng: np.random.Generator) -> dict[str, MarketData]:
    """target の時刻の並びに、pool の日付のブロックを全ペア同じ並びでつなぐ。足の 4 点の形と、その日付の金利差を運ぶ。
    スプレッドは target のまま。値の水準は target の最初の足の終値から始める。"""
    k = 4
    first = next(iter(target.values()))
    n = len(first.times) // k
    m = len(next(iter(pool.values())).times) // k
    if m < block + 2:
        raise ValueError("real_pairs_bootstrap: pool is shorter than a block")
    picks: list[int] = []
    while len(picks) < n - 1:
        st = int(rng.integers(1, m - block + 1))
        picks.extend(range(st, st + block))
    src = np.asarray(picks[:n - 1], dtype=np.int64)
    rates = load_rates(MULTI_RATES)
    out = {}
    for p, tg in target.items():
        pl = pool[p]
        lp = np.log(pl.mid).reshape(m, k)
        prev_close = lp[src - 1, -1]
        syn_close = np.log(tg.mid[k - 1]) + np.cumsum(lp[src, -1] - prev_close)
        base = np.r_[np.log(tg.mid[k - 1]), syn_close[:-1]]
        pts = base[:, None] + (lp[src] - prev_close[:, None])
        mid = np.r_[tg.mid[:k], np.exp(pts.ravel())]
        spread = tg.spread[:n * k]
        src_t = pl.times.reshape(m, k)[src, 0]
        diff = rates.pair_differential(p, src_t)
        diff0 = rates.pair_differential(p, tg.times[:1])
        meta = dict(tg.meta)
        meta["rate_diff_points"] = np.r_[np.repeat(diff0, k), np.repeat(diff, k)]
        out[p] = MarketData(times=tg.times[:n * k], mid=mid, spread=spread, pip=tg.pip, step_minutes=tg.step_minutes,
                            meta=meta, model_info=tg.model_info)
    return out


@MARKETS.register("real_pairs", version="1",
                  description="相場の DB の主要ペアを日付をそろえて 1 つの口座で回す(円に換算、記録のスプレッド、"
                              "ペアの金利差)",
                  tags=("real", "basket"))
def build_real_pairs(params: dict) -> RealPairsModel:
    return RealPairsModel(params)


@MARKETS.register("real_pairs_bootstrap", version="1",
                  description="real_pairs の日付のブロックを全ペア同じ並びでつないだ別の歴史(金利差も日付と一緒に運ぶ)",
                  tags=("real", "random", "basket"))
def build_real_pairs_bootstrap(params: dict) -> RealPairsBootstrap:
    return RealPairsBootstrap(params)
