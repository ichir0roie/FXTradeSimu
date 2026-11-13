"""実験の設定(YAML / dict)。

    name: long_run
    proposal: E001            # 任意。docs/proposals の ID
    years: 10
    seeds: [1, 2, 3, 4, 5]    # ランダム相場のパターン。実データの再生(real)は 1 本だけ回す
    detail_seeds: [1]         # ローソク足・取引・細かい資産曲線まで保存するシード
    workers: 4               # 省略で default_workers()(環境変数 FXSIM_WORKERS か、論理 CPU の半分 − 2)
    engine: {initial_balance: 1000000, units: 10000}
    storage: {candle_timeframe: 60, trades: detail, equity_timeframe: 1440, equity_detail_timeframe: 60}
    live: {flush_every: 500, delay_ms: 0}               # delay_ms > 0 で書き込みをゆっくりにする
    brokers: [fair, {preset: malicious, overrides: {hunt_prob: 0.5}, label: evil}]
    markets: [random_walk, {model: malicious, label: mal_strong, params: {stop_hunt: {prob: 0.12}}, seeds: [1, 2]}]
    strategies: [random_entry, {name: ma_cross, params: {fast: 10}, label: ma_fast, schedule: {decision_every: 4}}]
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from ..backtest.engine import EngineConfig
from ..backtest.schedule import TradingSchedule


def default_workers() -> int:
    """ワーカー数の既定。手元の端末が固まらないよう、CPU を使い切らない。

    環境変数 FXSIM_WORKERS があればそれ。無ければ論理 CPU の半分(ハイパースレッドなら物理コア数)から
    2 を引いた数(1〜8)。
    ワーカーは優先度も下げる(experiments.runner)。ワーカー数は結果に影響しない。
    """
    env = os.environ.get("FXSIM_WORKERS", "").strip()
    if env:
        return max(1, int(env))
    return max(1, min((os.cpu_count() or 2) // 2 - 2, 8))


@dataclass
class MarketSpec:
    model: str
    label: str
    params: dict = field(default_factory=dict)
    seeds: list[int] | None = None        # この市場だけシード(パターン)を変えるとき

    @classmethod
    def parse(cls, x: str | dict) -> MarketSpec:
        if isinstance(x, str):
            return cls(x, x)
        return cls(x["model"], x.get("label") or x["model"], x.get("params") or {}, x.get("seeds"))


@dataclass
class StrategySpec:
    name: str
    label: str
    params: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)   # TradingSchedule(判断の間隔・取引時間)

    @classmethod
    def parse(cls, x: str | dict) -> StrategySpec:
        if isinstance(x, str):
            return cls(x, x)
        return cls(x["name"], x.get("label") or x["name"], x.get("params") or {}, x.get("schedule") or {})

    def trading_schedule(self) -> TradingSchedule:
        return TradingSchedule.from_dict(self.schedule)


@dataclass
class BrokerSpec:
    preset: str
    label: str
    overrides: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, x: str | dict) -> BrokerSpec:
        if isinstance(x, str):
            return cls(x, x)
        return cls(x["preset"], x.get("label") or x["preset"], x.get("overrides") or {})


@dataclass
class StorageConfig:
    """どこまで細かく保存するか。DB を小さく保つため、詳細保存のシードとそれ以外で分ける。"""
    candle_timeframe: int = 60            # 詳細保存のローソク足(分)
    trades: str = "detail"                # detail(詳細保存のシードだけ)| all | none
    equity_timeframe: int = 1440          # 詳細保存しないパターンの資産曲線の間隔(分)
    equity_detail_timeframe: int | None = None   # 詳細保存のパターンの資産曲線の間隔(既定 = candle_timeframe)
    events: bool = True                   # 詳細保存のパターンの悪意イベントを保存するか

    def __post_init__(self):
        if self.trades not in ("detail", "all", "none"):
            raise ValueError("storage.trades must be detail / all / none")
        if self.equity_detail_timeframe is None:
            self.equity_detail_timeframe = self.candle_timeframe

    def store_trades(self, detail: bool) -> bool:
        return self.trades == "all" or (self.trades == "detail" and detail)


@dataclass
class LiveConfig:
    flush_every: int = 500     # 何本ごとに途中経過を書くか
    delay_ms: float = 0.0      # 書くたびに待つ時間(詳細保存のパターンだけ)


def _typed(cls, value):
    if isinstance(value, cls):
        return value
    value = dict(value or {})
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    return cls(**value)


@dataclass
class ExperimentConfig:
    name: str
    markets: list[MarketSpec]
    strategies: list[StrategySpec]
    brokers: list[BrokerSpec] = field(default_factory=lambda: [BrokerSpec("fair", "fair")])
    years: float = 3.0
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    detail_seeds: list[int] | None = None
    workers: int = 0
    kind: str = "experiment"
    proposal: str | None = None
    notes: str = ""
    engine: EngineConfig = field(default_factory=EngineConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def __post_init__(self):
        self.markets = [MarketSpec.parse(x) if not isinstance(x, MarketSpec) else x for x in self.markets]
        self.strategies = [StrategySpec.parse(x) if not isinstance(x, StrategySpec) else x for x in self.strategies]
        self.brokers = [BrokerSpec.parse(x) if not isinstance(x, BrokerSpec) else x for x in self.brokers]
        self.engine = _typed(EngineConfig, self.engine)
        self.storage = _typed(StorageConfig, self.storage)
        self.live = _typed(LiveConfig, self.live)
        if self.detail_seeds is None:
            self.detail_seeds = self.seeds[:1]
        if not self.workers:
            self.workers = default_workers()
        for kind, labels in (("market", [m.label for m in self.markets]),
                             ("strategy", [s.label for s in self.strategies]),
                             ("broker", [b.label for b in self.brokers])):
            if len(labels) != len(set(labels)):
                raise ValueError(f"{kind} labels must be unique: {labels}")

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)
