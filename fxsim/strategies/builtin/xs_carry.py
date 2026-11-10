"""S032 xs_carry: 通貨のキャリー(クロスセクション)。金利の高い通貨を買い、低い通貨を売る。

対円 7 ペアを「基軸通貨の金利 − 円の金利」で順位付けし、上位 n_long を買い・下位 n_short を売る
(hold_days 本ごとに入れ替え)。
金利は FRED の短期金利の月次(data/real/short_rates_monthly_fred.csv、月 M の平均は翌月 1 日から有効)。
儲けの源はスワップ(金利差)で、危機のときに高金利通貨がまとめて急落するのが弱み。
合成の相場(basket_bootstrap など)にも、実際の日付の金利をそのまま当てはめる。
"""
from dataclasses import dataclass

from .. import STRATEGIES
from ..basket import MULTI_RATES, CrossSectional, rate_differentials


@STRATEGIES.register("xs_carry", version="1", description="金利の高い通貨を買い、低い通貨を売る(対円 7 ペアの順位)")
class XsCarry(CrossSectional):
    name = "xs_carry"

    @dataclass
    class Params(CrossSectional.Params):
        rates_path: str = MULTI_RATES

    def score(self, bars_by_pair, c):
        return rate_differentials(self.pairs, bars_by_pair[self.pairs[0]].t, self.p.rates_path)
