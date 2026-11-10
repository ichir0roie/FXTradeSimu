"""複数の通貨ペアをまとめて作る市場モデル(バスケット)。日足だけ。

R009 basket_real         対円 7 ペアの実データ(FRED、終値だけ)をそのまま再生する
R010 basket_bootstrap    日次リターンを、全ペアで同じブロック(同じ日付の並び)で並べ替える。ペアどうしの相関は残る
R011 basket_placebo      日次リターンに、全ペアで同じ日ごとの乱数の符号(±1)を掛ける。相関・ボラの塊・大きさは残り、
                         方向の情報だけが無い(作り方の癖を測る偽薬)
M007 basket_random_walk  実データの日次リターンの共分散を持つ正規乱数(ドリフト 0)。完全ランダムの帰無

生成した MarketData は先頭のペア(既定 usdjpy)のもので、全ペアは `meta["legs"]`(ペア -> MarketData)に入る。
1 ペアの戦略はそのまま先頭のペアで回り、複数ペアの戦略(Strategy.multi)は全ペアで回る(backtest/portfolio.py)。
足の中の経路は持たない(日足の再生と同じく、始値 = 前日の終値の 2 点)。損切り・利確は足の高安で判定する。
スプレッドは国内の店頭業者の原則固定の幅を参考にした概算の固定値(実際には早朝・指標で広がる。広告の幅は最良時の値)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ...calendar import NS_DAY, NS_HOUR
from ...data import MarketData
from ...data.fred import FX_OUT, PAIRS
from ...paths import resolve
from ...registry import params_hash
from .. import MARKETS

DEFAULT_SPREAD_PIPS = {"usdjpy": 0.2, "eurjpy": 0.4, "gbpjpy": 0.9, "audjpy": 0.5, "nzdjpy": 1.2, "cadjpy": 1.5,
                       "chfjpy": 1.6}
PIP = 0.01


@dataclass
class BasketSource:
    path: str = FX_OUT
    start: str | None = None
    end: str | None = None
    block_days: float = 20.0       # bootstrap のブロックの長さ(日)
    # bootstrap: ペアごとの日次リターンの平均を引いてから並べ替える(この期間の上げ下げ = ドリフトを消し、
    # ブロックの中のトレンド・ボラの塊・相関だけを残す)。ドリフトを覚えただけの勝ちと、本物の時系列の癖を切り分ける
    demean: bool = False


@dataclass
class BasketParams:
    pairs: list[str] = field(default_factory=lambda: list(PAIRS))
    spread_pips: dict = field(default_factory=dict)       # ペア -> pip(既定は DEFAULT_SPREAD_PIPS)


@lru_cache(maxsize=4)
def load_wide(path: str) -> pd.DataFrame:
    df = pd.read_csv(resolve(path))
    t = pd.to_datetime(df.pop("time")).values.astype("datetime64[ns]").astype(np.int64) + 22 * NS_HOUR
    df.index = t                                           # 足の終わり = その日の NY クローズ(22:00 UTC)
    return df


class BasketModel:
    def __init__(self, name: str, version: str, mode: str, params: dict | None = None):
        params = dict(params or {})
        self.name, self.version, self.mode = name, version, mode
        self.src = BasketSource(**(params.pop("source", None) or {}))
        self.p = BasketParams(**params)
        unknown = [x for x in self.p.pairs if x not in PAIRS]
        if unknown:
            raise KeyError(f"{name}: unknown pairs {unknown} (known: {list(PAIRS)})")

    @property
    def deterministic(self) -> bool:
        return self.mode == "replay"

    @property
    def source(self):
        return self.src

    def info(self) -> dict:
        params = {"source": asdict(self.src), **asdict(self.p)}
        return {"name": self.name, "version": self.version, "components": ["basket", self.mode], "params": params,
                "params_hash": params_hash(params)}

    def window(self, years: float | None, days: float | None) -> pd.DataFrame:
        """使う期間の実データ(日 x ペア)。"""
        df = load_wide(self.src.path)[self.p.pairs].dropna()
        t = df.index.to_numpy()
        lo, hi = 0, len(t)
        if self.src.start:
            lo = int(np.searchsorted(t, np.datetime64(self.src.start, "ns").astype(np.int64)))
        if self.src.end:
            hi = int(np.searchsorted(t, np.datetime64(self.src.end, "ns").astype(np.int64) + NS_DAY, "right"))
        if days is not None and years is None:
            years = days / 365.25
        if years is not None and not self.src.start:
            lo = max(lo, int(np.searchsorted(t, t[hi - 1] - int(years * 365.25 * NS_DAY))))
        w = df.iloc[lo:hi]
        if len(w) < 3:
            raise ValueError(f"{self.name}: data window too short ({len(w)} rows)")
        return w

    def _paths(self, logc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(日 x ペア)の log 終値。"""
        r = np.diff(logc, axis=0)
        if self.mode == "replay":
            return logc
        if self.mode == "bootstrap":
            m = len(r)
            k = max(1, int(round(self.src.block_days)))
            picks: list[int] = []
            while len(picks) < m:
                st = int(rng.integers(0, max(1, m - k)))
                picks.extend(range(st, min(m, st + k)))
            src = r - r.mean(axis=0) if self.src.demean else r
            nr = src[np.asarray(picks[:m])]
        elif self.mode == "placebo":
            nr = r * np.where(rng.random(len(r)) < 0.5, -1.0, 1.0)[:, None]
        elif self.mode == "random_walk":
            cov = np.cov(r, rowvar=False)
            chol = np.linalg.cholesky(cov + 1e-14 * np.eye(len(cov)))
            nr = rng.standard_normal(r.shape) @ chol.T
        else:
            raise ValueError(f"unknown basket mode {self.mode}")
        return np.vstack([logc[:1], logc[:1] + np.cumsum(nr, axis=0)])

    def generate(self, *, years: float | None = None, days: float | None = None, seed: int = 0) -> MarketData:
        rng = np.random.default_rng([seed, 4242])
        w = self.window(years, days)
        t_end = w.index.to_numpy().astype(np.int64)
        prices = np.exp(self._paths(np.log(w.to_numpy(float)), rng))
        sec = 1_000_000_000
        start = t_end - NS_DAY
        times = np.stack([start + sec, t_end - sec], axis=1).ravel()
        info = self.info()
        legs = {}
        for j, pair in enumerate(self.p.pairs):
            c = prices[:, j]
            o = np.r_[c[0], c[:-1]]                        # 終値だけのデータの始値は前日の終値
            mid = np.stack([o, c], axis=1).ravel()
            sp = float(self.p.spread_pips.get(pair, DEFAULT_SPREAD_PIPS[pair])) * PIP
            legs[pair] = MarketData(times=times, mid=mid, spread=np.full(len(mid), sp), pip=PIP, step_minutes=1440,
                                    meta={"pair": pair}, model_info={**info, "pair": pair})
        first = legs[self.p.pairs[0]]
        first.meta["legs"] = legs
        first.meta["real_source"] = f"{self.src.path} ({self.mode})"
        return first


def _register(name: str, version: str, mode: str, description: str, tags: tuple[str, ...]):
    @MARKETS.register(name, version=version, description=description, tags=tags)
    def build(params: dict) -> BasketModel:
        return BasketModel(name, version, mode, params)

    return build


_register("basket_real", "1", "replay", "対円 7 ペアの実データ(FRED 日足)をそのまま再生", ("real", "basket"))
_register("basket_bootstrap", "1", "bootstrap", "対円 7 ペアの日次リターンを全ペア同じブロックで並べ替え",
          ("real", "random", "basket"))
_register("basket_placebo", "1", "placebo", "対円 7 ペアの日次リターンに全ペア共通の乱数の符号を掛けた偽薬",
          ("real", "random", "placebo", "basket"))
_register("basket_random_walk", "1", "random_walk", "実データの共分散を持つ正規乱数の 7 ペア(ドリフト 0)",
          ("random", "basket"))
