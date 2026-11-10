"""M003/M004 malicious: stylized に「市場の悪意」を足したもの。

- crowd_squeeze  群衆(順張り/逆張り)の逆を突く圧力
- false_breakout レンジ抜けのダマシ
- stop_hunt      逆指値が溜まる水準へのヒゲ
部品は params で個別に外せる(例: {"stop_hunt": None})。

v2(M004): 悪意が「勝っている側」に適応する。群衆は直近で儲かっている側へ群がり(chase_gain)、
ダマシとストップ狩りは直近の結末と逆へ寄る(adapt_gain)。v1 では 10 年 x 8 パターンで
ブレイクアウトが素のシャープ t = 5.4 で勝っていた。v1 を再現するには
{"crowd_squeeze": {"chase_gain": 0}, "false_breakout": {"adapt_gain": 0}, "stop_hunt": {"adapt_gain": 0}}。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import CrowdSqueeze, DynamicSpread, FalseBreakout, StopHunt
from .stylized import stylized_components


@MARKETS.register("malicious", version="2",
                  description="stylized + 群衆の逆突き・ダマシ・ストップ狩り", tags=("malice",))
def build(params: dict) -> MarketModel:
    comps = [c for c in stylized_components() if not isinstance(c, DynamicSpread)]
    comps += [CrowdSqueeze(chase_gain=1.0), FalseBreakout(adapt_gain=1.0), StopHunt(adapt_gain=1.0), DynamicSpread()]
    return MarketModel("malicious", "2", comps).with_params(params)
