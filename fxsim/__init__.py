"""FXTradeSimu: 「悪意ある市場」を合成し、FX 戦略を長期シミュレーションするためのパッケージ。

主な入口:
    fxsim.catalog          組み込みの市場モデル・戦略をすべて登録する
    fxsim.markets          市場モデル(部品の組み合わせ)
    fxsim.strategies       戦略
    fxsim.backtest         約定・損益の計算
    fxsim.experiments      実験(市場 x ブローカー x 戦略 x シード)の実行
    fxsim.storage          SQLite への保存(Web UI が読む)
"""

__version__ = "0.2.0"
