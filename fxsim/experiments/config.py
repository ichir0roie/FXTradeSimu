"""実験の設定(YAML / dict)。

    name: long_run
    proposal: E001            # 任意。docs/proposals の ID
    years: 10
    seeds: [1, 2, 3, 4, 5]    # ランダム相場のパターン。実データの再生(real)は 1 本だけ回す
    detail_seeds: [1]         # ローソク足・取引・細かい資産曲線まで保存するシード
    workers: 4
    engine: {initial_balance: 1000000, units: 10000}
    storage: {candle_timeframe: 60, trades: detail, equity_timeframe: 1440, equity_detail_timeframe: 60}
                              # trades: detail | all | none。資産曲線は詳細保存のパターンとそれ以外で間隔を分けられる
    live: {flush_every: 500, delay_ms: 0}               # delay_ms > 0 で画面で追える速さに落とす
    brokers: [fair, {preset: malicious, overrides: {hunt_prob: 0.5}, label: evil}]
    markets: [random_walk, {model: malicious, label: mal_strong, params: {stop_hunt: {prob: 0.12}}}]
    strategies: [random_entry, {name: ma_cross, params: {fast: 10}, label: ma_fast}]
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class MarketSpec:
    model: str
    label: str
    params: dict = field(default_factory=dict)
    seeds: list[int] | None = None        # この市場だけシード(パターン)を変えるとき


@dataclass
class StrategySpec:
    name: str
    label: str
    params: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)   # TradingSchedule(判断の間隔・取引時間)


@dataclass
class BrokerSpec:
    preset: str
    label: str
    overrides: dict = field(default_factory=dict)


def _market(x) -> MarketSpec:
    if isinstance(x, str):
        return MarketSpec(x, x)
    return MarketSpec(x["model"], x.get("label") or x["model"], x.get("params") or {}, x.get("seeds"))


def _strategy(x) -> StrategySpec:
    if isinstance(x, str):
        return StrategySpec(x, x)
    return StrategySpec(x["name"], x.get("label") or x["name"], x.get("params") or {}, x.get("schedule") or {})


def _broker(x) -> BrokerSpec:
    if isinstance(x, str):
        return BrokerSpec(x, x)
    return BrokerSpec(x["preset"], x.get("label") or x["preset"], x.get("overrides") or {})


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
    engine: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.detail_seeds is None:
            self.detail_seeds = self.seeds[:1]
        if not self.workers:
            self.workers = max(1, min(os.cpu_count() or 1, 8))
        # equity_timeframe: 詳細保存しないパターンの資産曲線の間隔(分)。detail は詳細保存のパターン用(既定は足と同じ)
        self.storage = {"candle_timeframe": 60, "trades": "detail", "equity_timeframe": 1440, **self.storage}
        self.storage.setdefault("equity_detail_timeframe", self.storage["candle_timeframe"])
        self.live = {"flush_every": 500, "delay_ms": 0, **self.live}
        labels = [m.label for m in self.markets]
        if len(labels) != len(set(labels)):
            raise ValueError(f"market labels must be unique: {labels}")
        labels = [s.label for s in self.strategies]
        if len(labels) != len(set(labels)):
            raise ValueError(f"strategy labels must be unique: {labels}")

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        d["markets"] = [_market(x) for x in d.get("markets", [])]
        d["strategies"] = [_strategy(x) for x in d.get("strategies", [])]
        if "brokers" in d:
            d["brokers"] = [_broker(x) for x in d["brokers"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)
