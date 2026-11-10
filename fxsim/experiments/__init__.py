"""実験: 市場 x ブローカー x 戦略 x シードの組を回し、結果を SQLite に記録する。

    config.py   実験の設定(YAML / dict)
    runner.py   計画と実行(経路ごとに 1 プロセス)
    stats.py    パターンごとの統計(平均・中央値・合計・t 値など)
"""
from .config import BrokerSpec, ExperimentConfig, LiveConfig, MarketSpec, StorageConfig, StrategySpec
from .runner import run_experiment

__all__ = ["BrokerSpec", "ExperimentConfig", "LiveConfig", "MarketSpec", "StorageConfig", "StrategySpec",
           "run_experiment"]
