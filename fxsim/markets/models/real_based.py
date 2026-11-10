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
v3(R007): 足の中に薄商いの期間(HolidayLiquidity)と、終値の手前で戻り切るフラッシュクラッシュ
(FlashCrashOverlay)を足した。日足の終値は変わらない。v2 を再現するには {"holidays": None, "flash_overlay": None}。
v4(R008 / E007): 足の中の振れ幅を「直前の局所ボラを中心にした事前分布」と「その足のリターン」から事後分布で引き
(calibrate_vol: posterior)、終値へのずれを事前の分散の時計で配る(clock: expected)。v3 は足の中の荒れ方から
日足のリターンの大きさが漏れ、1 時間足の RSI・ストキャスの逆張りが素で年 +800 pip 勝っていた。
あわせて NewsShocks の並べ替えを安定ソートに直した(同じシードでも CPU によって相場が変わっていた)。
v3 の合わせ方は {"anchor": {"calibrate_vol": "local", "clock": "variance"}}(値動きそのものは再現しない)。
v5(R009): 足の中の振れ幅の事前分布の中心を、直前 21 本と 250 本の r² の平均の半々にした(prior_long_weight 0.5)。
直前 21 本だけでは静かな時期の後に振れ幅が小さすぎ、終値への補正が日中のトレンドになっていた
(偽薬で 1 時間足のブレイクアウトが素で +1.2 pip / 取引、t 6)。v4 は {"anchor": {"prior_long_weight": 0}}。
データの差し替えは params の source(例: {"source": {"path": "data/real/xxx.csv"}})。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import (
    ConstantSpread,
    DynamicSpread,
    FatTails,
    FlashCrashOverlay,
    HolidayLiquidity,
    NewsShocks,
    SessionSeasonality,
    StochasticVolatility,
    StopHunt,
)
from ..components.real import AnchorBridge, RealReplay, RealSkeleton


def _intraday():
    """骨格の足の中を作る部品(局面・窓は骨格が持つので入れない)。"""
    return [SessionSeasonality(), StochasticVolatility(), HolidayLiquidity(), NewsShocks(), FatTails()]


def _bridge(skeleton: RealSkeleton, *malice) -> list:
    """骨格 + 足の中の合成 + (悪意) + 終値への合わせ込み + 戻り切るフラッシュクラッシュ + スプレッド。"""
    return [skeleton, *_intraday(), *malice, AnchorBridge(), FlashCrashOverlay(), DynamicSpread()]


@MARKETS.register("real", version="2", description="実データをそのまま再生(既定: USDJPY 日足 FRED)",
                  tags=("real",))
def build_real(params: dict) -> MarketModel:
    return MarketModel("real", "2", [RealReplay(), ConstantSpread()]).with_params(params)


@MARKETS.register("real_bridge", version="5",
                  description="実データの足を骨格に、足の中の動きをシードごとにランダム合成", tags=("real", "random"))
def build_bridge(params: dict) -> MarketModel:
    return MarketModel("real_bridge", "5", _bridge(RealSkeleton(mode="replay"))).with_params(params)


@MARKETS.register("real_bridge_malicious", version="5",
                  description="real_bridge + 適応するストップ狩り(ストレス試験)", tags=("real", "random", "malice"))
def build_bridge_malicious(params: dict) -> MarketModel:
    comps = _bridge(RealSkeleton(mode="replay"), StopHunt(adapt_gain=1.0))
    return MarketModel("real_bridge_malicious", "5", comps).with_params(params)


@MARKETS.register("real_bootstrap", version="5",
                  description="実データのリターンをブロック単位で並べ替えた骨格 + 足の中の合成",
                  tags=("real", "random"))
def build_bootstrap(params: dict) -> MarketModel:
    return MarketModel("real_bootstrap", "5", _bridge(RealSkeleton(mode="bootstrap"))).with_params(params)


@MARKETS.register("real_bridge_placebo", version="5",
                  description="実データのリターンの符号だけでたらめにした骨格 + 同じ合成(作り方の癖を測る偽薬)",
                  tags=("real", "random", "placebo"))
def build_placebo(params: dict) -> MarketModel:
    return MarketModel("real_bridge_placebo", "5", _bridge(RealSkeleton(mode="placebo"))).with_params(params)
