"""S034 xs_value: 通貨のバリュー(クロスセクション)。長い目で見て割安な通貨を買い、割高な通貨を売る。

割安さの代わりに、lookback 本(既定 5 年)の log リターンの符号を反転したもの(長期の反転)を使う。
本来のバリューは物価で調整した実質の為替(購買力平価)で測るが、物価のデータを持っていないので名目の代用。
物価の高い国の通貨は名目で下がり続けやすいので、それを「割安」と取り違える偏りがある(提案書 S032-S036 を参照)。
"""
from dataclasses import dataclass

from .. import STRATEGIES
from ..basket import CrossSectional, lagged_log_return


@STRATEGIES.register("xs_value", version="1",
                     description="過去 5 年で下がった通貨を買い、上がった通貨を売る(長期の反転)")
class XsValue(CrossSectional):
    name = "xs_value"

    @dataclass
    class Params(CrossSectional.Params):
        lookback: int = 1260
        skip: int = 0

    def score(self, bars_by_pair, c):
        return -lagged_log_return(c, self.p.lookback, self.p.skip)
