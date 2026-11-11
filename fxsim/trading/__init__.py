"""MT5 の口座で、動作ケース(cases/behaviors)の戦略を常駐して動かす。

流れ(足が 1 本確定するたびに 1 回):
    1. MT5 から足を取る(起点 anchor から、いま作られている足まで)                       feed.py / mt5.py
    2. その足でバックテスト(Backtester)を回し、最後の確定足で戦略が出した注文と、
       その時点のシミュレーション上の建玉を取り出す                                   signals.py
    3. 実口座の建玉と突き合わせて、出す注文を決める                                     trader.py(plan)
    4. MT5 に注文を出し、状態と記録を data/trading/<name>/ に残す                       trader.py / service.py
戦略のコードはバックテストと同じものをそのまま使う。書式と立ち上げ方は docs/TRADING.md。
"""
from .config import TraderConfig, load_trader_config
from .signals import SimSnapshot, simulate

__all__ = ["TraderConfig", "load_trader_config", "SimSnapshot", "simulate"]
