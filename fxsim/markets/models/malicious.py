"""M003 malicious: stylized に「市場の悪意」を足したもの。

- crowd_squeeze  群衆(順張り/逆張り)の逆を突く圧力
- false_breakout レンジ抜けのダマシ
- stop_hunt      逆指値が溜まる水準へのヒゲ
部品は params で個別に外せる(例: {"stop_hunt": None})。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import CrowdSqueeze, DynamicSpread, FalseBreakout, StopHunt
from .stylized import stylized_components


@MARKETS.register("malicious", version="1",
                  description="stylized + 群衆の逆突き・ダマシ・ストップ狩り", tags=("malice",))
def build(params: dict) -> MarketModel:
    comps = [c for c in stylized_components() if not isinstance(c, DynamicSpread)]
    comps += [CrowdSqueeze(), FalseBreakout(), StopHunt(), DynamicSpread()]
    return MarketModel("malicious", "1", comps).with_params(params)
