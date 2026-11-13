"""戦略を動かすランナー。どの戦略(strategies/ の 1 ファイル 1 クラス)も、同じ窓口で
シミュレーションと MT5 の口座の両方で動く。

    Runner        ベース。動作ケース(cases/behaviors)か戦略の名前から、戦略・パラメータ・ブローカー・エンジン・
                  取引時間を決め、戦略を作る。最後の確定足の判断(コアのモデル、decide)を共通で持つ
    SimRunner     シミュレーション(バックテスト)。相場モデルの経路の上で戦略を最後まで回し、成績を返す
    Mt5Runner     MT5 の口座での常駐。足が確定するたびに decide で判断し、実口座の建玉と突き合わせて注文を出す
                  (1 銘柄は LiveTrader、複数ペアは PortfolioTrader)
    ReplayRunner  Mt5Runner の端末を、相場の DB の 1 分足を再生する偽の端末に替えたもの(故障も入れられる)

例:
    SimRunner("B047").run("real_pairs", params={"source": {"start": "2020-01-01"}})
    Mt5Runner("configs/trading/demo-majors-b047.yaml").run()
    ReplayRunner(["configs/trading/demo-majors-b047.yaml"]).run()
"""
from .base import Runner
from .mt5 import Mt5Runner
from .replay import ReplayRunner
from .sim import SimResult, SimRunner

__all__ = ["Mt5Runner", "ReplayRunner", "Runner", "SimResult", "SimRunner"]
