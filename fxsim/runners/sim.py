"""シミュレーション(バックテスト)のランナー。"""
from __future__ import annotations

from dataclasses import dataclass

from ..backtest.metrics import compute_metrics
from ..backtest.portfolio import build_backtester
from ..data import MarketData
from ..markets import MARKETS
from .base import Runner


@dataclass
class SimResult:
    backtester: object           # Backtester / PortfolioBacktester / Mt5Tester(取引・資産曲線・費用の明細を持つ)
    metrics: dict                # compute_metrics の成績

    @property
    def trades(self):
        return self.backtester.trades


class SimRunner(Runner):
    """相場モデルの経路の上で、戦略を最後まで回す。エンジンは戦略の種類で選ぶ(1 ペア・複数ペア・MT5 のテスター)。"""

    def backtester(self, md: MarketData, seed: int = 1):
        return build_backtester(md, self.make_strategy(), self.broker_config(), self.engine_config, seed=seed,
                                schedule=self.schedule)

    def backtest(self, md: MarketData, seed: int = 1, progress=None, every: int = 500) -> SimResult:
        bt = self.backtester(md, seed)
        bt.run(progress=progress, every=every)
        return SimResult(bt, compute_metrics(bt))

    def run(self, market: str | MarketData, *, params: dict | None = None, years: float | None = None,
            days: float | None = None, seed: int = 1) -> SimResult:
        """market は市場モデルの名前(params で上書き)か、作った MarketData。"""
        md = market if isinstance(market, MarketData) else \
            MARKETS.create(market, params or {}).generate(years=years, days=days, seed=seed)
        return self.backtest(md, seed)


__all__ = ["SimResult", "SimRunner"]
