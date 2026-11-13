"""MT5 の口座での常駐のランナー(fxsim trade)。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

from ..backtest.broker import BrokerConfig
from ..cases.model import BehaviorCase
from ..strategies import STRATEGIES
from ..strategies.main.base import MainStrategy
from ..trading.config import TraderConfig, load_trader_config
from .base import Runner


class Mt5Runner(Runner):
    """足が確定するたびに decide(コアのモデル)で判断し、実口座の建玉と突き合わせて注文を出す。

    設定(configs/trading/*.yaml)の behavior の戦略を、strategy_params で上書きして動かす。1 銘柄の戦略は LiveTrader、
    複数ペアの戦略(symbols のある設定)は PortfolioTrader が実口座とのやり取りを受け持つ。
    シミュレーションの中の口座の床は切る(床は実口座の有効証拠金で見るので)。シミュレーションのブローカーは
    trader.sim_broker_for(約定は素直、スワップの決め方は動作ケースのもの)。
    """

    def __init__(self, config: TraderConfig | str | Path, *, behavior: BehaviorCase | None = None,
                 strategy_factory: Callable[[], object] | None = None, module=None,
                 sleep: Callable[[float], None] | None = None):
        cfg = config if isinstance(config, TraderConfig) else load_trader_config(config)
        super().__init__(behavior or cfg.behavior, strategy_params=cfg.strategy_params,
                         strategy_factory=strategy_factory)
        self.cfg, self.module, self.sleep = cfg, module, sleep
        self.config_path = None if isinstance(config, TraderConfig) else str(config)

    def make_strategy(self):
        if self._factory is not None:
            return self._factory()
        params = dict(self.params)
        probe = STRATEGIES.create(self.strategy_name, params)
        if isinstance(probe, MainStrategy):
            params["guard"] = {**(params.get("guard") or {}), "protect": 0.0}
        elif hasattr(probe.p, "protect"):
            params["protect"] = 0.0
        return STRATEGIES.create(self.strategy_name, params)

    @cached_property
    def sim_broker(self) -> BrokerConfig:
        from ..trading.trader import sim_broker_for
        return sim_broker_for(self.behavior, self.cfg.sim_broker, self.engine_config, self.mt5_engine)

    def make_trader(self, client, expected_login: int | None = None):
        from ..trading.portfolio import PortfolioTrader
        from ..trading.trader import LiveTrader
        cls = PortfolioTrader if self.cfg.symbols else LiveTrader
        return cls(self.cfg, client, runner=self, sleep=self.sleep, expected_login=expected_login)

    def service(self):
        from ..trading.service import Service
        return Service(self.cfg, module=self.module, runner=self)

    def run(self, once: bool = False) -> None:
        svc = self.service()
        svc.install_signals()
        svc.run(once=once)

    def check(self) -> dict:
        """端末につなぎ、いまの足での判断を返す(注文は出さない)。"""
        self.cfg.dry_run = True
        svc = self.service()
        trader = svc.connect()
        c = svc.client
        try:
            forming = int(c.rates(self.cfg.symbol, trader.timeframe, 1)["time"][-1])
            if self.cfg.symbols:
                decided = trader.decide(forming)
                if decided is None:
                    return {"waiting": "some symbols have no forming bar yet"}
                snap, per, meta = decided
                return {"meta": meta, "sim_equity": snap.equity, "server_hour": trader.server_hour(),
                        "fill_server_hour": self.cfg.fill_server_hour,
                        "pairs": {s: {"sim_position": asdict(snap.pairs[s[:6].lower()].position)
                                      if snap.pairs[s[:6].lower()].position else None,
                                      "real": [asdict(p) for p in per[s][0]],
                                      "actions": [asdict(a) for a in per[s][1]]} for s in self.cfg.symbols}}
            snap, real, actions, meta = trader.decide(forming)
            return {"meta": meta, "sim_position": asdict(snap.position) if snap.position else None,
                    "sim_orders": [asdict(o) for o in snap.orders], "real": [asdict(p) for p in real],
                    "actions": [asdict(a) for a in actions]}
        finally:
            c.shutdown()


__all__ = ["Mt5Runner"]
