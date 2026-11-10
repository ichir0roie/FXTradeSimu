"""実際の相場を元にした市場モデル。

R001 real                  実データをそのまま再生(シードで変わらない)
R002 real_bridge           実データの足(既定: USDJPY 日足)を骨格に、足の中をランダムに合成
R003 real_bridge_malicious R002 + ストップ狩り(適応つき)。ブローカーとストップ狩りへの耐性を見るストレス試験。
                           日足の終値が実データに固定されるため、持続するずれを作る悪意(群衆の逆突き・ダマシ)は
                           その日のうちに戻る動きに変わり、日中の逆張りの穴になる。v2 ではそれらを外した
R004 real_bootstrap        実データのリターンをブロック単位で並べ替えた骨格 + 足の中の合成
R006 real_bridge_placebo   実データのリターンの符号だけでたらめにした骨格 + 同じ合成(作り方の癖を測る偽薬)

v2(R005): 終値への補正を「動いた量」に比例して配り、足ごとの振れ幅を実データの局所ボラと
その足のリターンの混ぜ合わせに合わせる(day_weight 0.6)。
v1 は補正を時刻に比例して配っていたため、日中の平均回帰(1 時間足の逆張りが勝つ穴)があった。
v1 を再現するには params に {"anchor": {"calibrate_vol": "global", "clock": "time"}}。
データの差し替えは params の source(例: {"source": {"path": "data/real/xxx.csv"}})。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import (
    ConstantSpread,
    DynamicSpread,
    FatTails,
    NewsShocks,
    SessionSeasonality,
    StochasticVolatility,
    StopHunt,
)
from ..components.real import AnchorBridge, RealReplay, RealSkeleton


def _intraday():
    """骨格の足の中を作る部品(局面・窓は骨格が持つので入れない)。"""
    return [SessionSeasonality(), StochasticVolatility(), NewsShocks(), FatTails()]


@MARKETS.register("real", version="1", description="実データをそのまま再生(既定: USDJPY 日足 FRED)",
                  tags=("real",))
def build_real(params: dict) -> MarketModel:
    return MarketModel("real", "1", [RealReplay(), ConstantSpread()]).with_params(params)


@MARKETS.register("real_bridge", version="2",
                  description="実データの足を骨格に、足の中の動きをシードごとにランダム合成", tags=("real", "random"))
def build_bridge(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="replay"), *_intraday(), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bridge", "2", comps).with_params(params)


@MARKETS.register("real_bridge_malicious", version="2",
                  description="real_bridge + 適応するストップ狩り(ストレス試験)", tags=("real", "random", "malice"))
def build_bridge_malicious(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="replay"), *_intraday(), StopHunt(adapt_gain=1.0), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bridge_malicious", "2", comps).with_params(params)


@MARKETS.register("real_bootstrap", version="2",
                  description="実データのリターンをブロック単位で並べ替えた骨格 + 足の中の合成",
                  tags=("real", "random"))
def build_bootstrap(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="bootstrap"), *_intraday(), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bootstrap", "2", comps).with_params(params)


@MARKETS.register("real_bridge_placebo", version="2",
                  description="実データのリターンの符号だけでたらめにした骨格 + 同じ合成(作り方の癖を測る偽薬)",
                  tags=("real", "random", "placebo"))
def build_placebo(params: dict) -> MarketModel:
    comps = [RealSkeleton(mode="placebo"), *_intraday(), AnchorBridge(), DynamicSpread()]
    return MarketModel("real_bridge_placebo", "2", comps).with_params(params)
