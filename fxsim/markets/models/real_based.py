"""実際の相場を元にした市場モデル。

R001 real                  実データをそのまま再生(シードで変わらない)
R002 real_bridge           実データの足(既定: USDJPY 日足)を骨格に、足の中をランダムに合成
R003 real_bridge_malicious R002 + 悪意(群衆の逆突き・ダマシ・ストップ狩り)
R004 real_bootstrap        実データのリターンをブロック単位で並べ替えた骨格 + 足の中の合成
データの差し替えは params の source(例: {"source": {"path": "data/real/xxx.csv"}})。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import (ConstantSpread, CrowdSqueeze, DynamicSpread, FalseBreakout, FatTails, NewsShocks,
                          SessionSeasonality, StochasticVolatility, StopHunt)
from ..components.real import AnchorBridge, RealReplay, RealSkeleton


def _intraday():
    """骨格の足の中を作る部品(局面・窓は骨格が持つので入れない)。"""
    return [SessionSeasonality(), StochasticVolatility(), NewsShocks(), FatTails()]


@MARKETS.register("real", version="1", description="実データをそのまま再生(既定: USDJPY 日足 FRED)",
                  tags=("real",))
def build_real(params: dict) -> MarketModel:
    return MarketModel("real", "1", [RealReplay(), ConstantSpread()]).with_params(params)


@MARKETS.register("real_bridge", version="1",
                  description="実データの足を骨格に、足の中の動きをシードごとにランダム合成", tags=("real", "random"))
def build_bridge(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="replay"), *_intraday(), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bridge", "1", comps).with_params(params)


@MARKETS.register("real_bridge_malicious", version="1",
                  description="real_bridge + 群衆の逆突き・ダマシ・ストップ狩り", tags=("real", "random", "malice"))
def build_bridge_malicious(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="replay"), *_intraday(), CrowdSqueeze(), FalseBreakout(), StopHunt(),
             AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bridge_malicious", "1", comps).with_params(params)


@MARKETS.register("real_bootstrap", version="1",
                  description="実データのリターンをブロック単位で並べ替えた骨格 + 足の中の合成", tags=("real", "random"))
def build_bootstrap(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="bootstrap"), *_intraday(), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bootstrap", "1", comps).with_params(params)
