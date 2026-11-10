"""S033 xs_momentum: 通貨のモメンタム(クロスセクション)。過去によく上がった通貨を買い、下がった通貨を売る。

対円 7 ペアを、lookback 本前から skip 本前までの log リターンで順位付けする(既定は 3 か月、直近を飛ばさない)。
1 ペアの時系列版(tsmom)と違い、通貨どうしの相対の強さだけを使う(円全体の上げ下げは打ち消す)。
"""
from dataclasses import dataclass

from .. import STRATEGIES
from ..basket import CrossSectional, lagged_log_return


@STRATEGIES.register("xs_momentum", version="1", description="過去 lookback 日に強かった通貨を買い、弱かった通貨を売る")
class XsMomentum(CrossSectional):
    name = "xs_momentum"

    @dataclass
    class Params(CrossSectional.Params):
        lookback: int = 63
        skip: int = 0

    def score(self, bars_by_pair, c):
        return lagged_log_return(c, self.p.lookback, self.p.skip)
