"""ランナーのベース: 動かす戦略の決め方と、最後の確定足の判断(コアのモデル)。"""
from __future__ import annotations

from collections.abc import Callable
from functools import cached_property

from ..backtest.broker import BrokerConfig
from ..backtest.engine import EngineConfig
from ..backtest.schedule import TradingSchedule
from ..cases.loader import load_cases
from ..cases.model import BehaviorCase
from ..data import Bars
from ..strategies import STRATEGIES


class Runner:
    """戦略を動かす土台。

    何を動かすかは、動作ケース(B047 などの ID か BehaviorCase)か、戦略の名前で渡す。どちらでも、パラメータ・
    ブローカー・エンジン・取引時間を上書きできる(上書きは動作ケースの値に重ねる)。
    戦略そのものは strategies/ の 1 ファイル 1 クラス(Strategy / BasketStrategy / MainStrategy のサブクラス)で、
    ランナーはその種類(1 ペアか複数ペアか、MT5 のテスターで回すか)を見て、合うエンジンを選ぶ。
    """

    def __init__(self, behavior: str | BehaviorCase | None = None, *, strategy: str | None = None,
                 strategy_params: dict | None = None, broker: str | dict | None = None,
                 engine: dict | None = None, schedule: dict | None = None,
                 strategy_factory: Callable[[], object] | None = None):
        if behavior is None and strategy is None and strategy_factory is None:
            raise ValueError("give a behavior, a strategy name or a strategy factory")
        if isinstance(behavior, str):
            behavior = load_cases().behaviors[behavior]
        self.behavior: BehaviorCase | None = behavior
        self.strategy_name = strategy or (behavior.strategy if behavior else None)
        base = behavior.strategy_params() if behavior else {}
        self.params = {**base, **(strategy_params or {})}
        if broker is None:
            self.broker_spec = behavior.broker_spec if behavior else {"preset": "fair", "overrides": {}}
        else:
            b = broker if isinstance(broker, dict) else {"preset": broker}
            self.broker_spec = {"preset": b["preset"], "overrides": b.get("overrides", {})}
        self.engine_dict = {**(behavior.engine if behavior else {}), **(engine or {})}
        self.schedule_dict = schedule if schedule is not None else (behavior.schedule if behavior else {})
        self._factory = strategy_factory

    # ---- 戦略 ----
    def make_strategy(self):
        """戦略のインスタンスを新しく作る(戦略は中に状態を持つので、回すたびに作り直す)。"""
        if self._factory is not None:
            return self._factory()
        return STRATEGIES.create(self.strategy_name, self.params)

    @cached_property
    def probe(self):
        """種類を調べるためのインスタンス(回さない)。"""
        return self.make_strategy()

    @property
    def timeframe(self) -> int:
        return int(self.probe.timeframe)

    @property
    def multi(self) -> bool:
        return bool(getattr(self.probe, "multi", False))

    @property
    def mt5_engine(self) -> bool:
        return getattr(self.probe, "engine", "classic") == "mt5"

    # ---- 口座の条件 ----
    def broker_config(self) -> BrokerConfig:
        return BrokerConfig.preset(self.broker_spec["preset"], self.broker_spec.get("overrides") or {})

    @cached_property
    def engine_config(self) -> EngineConfig:
        return EngineConfig(**self.engine_dict)

    @cached_property
    def schedule(self) -> TradingSchedule:
        return TradingSchedule.from_dict(self.schedule_dict)

    # ---- コアのモデル ----
    def decide(self, bars: Bars | dict[str, Bars], *, broker: BrokerConfig | None = None, pip: float | None = None,
               pips: dict[str, float] | None = None, seed: int = 0, strategy=None):
        """足(最後の 1 本は作りかけ)で戦略を回し、最後の確定足の判断(注文とその時点の建玉)を返す。
        1 ペアは SimSnapshot、複数ペア(bars がペア -> Bars)は PortfolioSnapshot。
        シミュレーションと MT5 の口座の両方が、これで同じ判断を得る。"""
        from ..trading.signals import simulate, simulate_portfolio

        broker = broker or self.broker_config()
        strategy = strategy or self.make_strategy()
        if isinstance(bars, dict):
            return simulate_portfolio(bars, strategy, broker, self.engine_config, self.schedule, pips or {}, seed)
        return simulate(bars, strategy, broker, self.engine_config, self.schedule, pip or 0.01, seed)

    def run(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


__all__ = ["Runner"]
