"""S011 gotobi: 五十日(ごとうび)の仲値トレード(5 分足)。

日本では 5・10 日と月末(休日なら前の営業日)に、輸入企業のドル買いが東京の仲値(9:55 JST)に集まり、
朝のうちはドル円が上がりやすい、と言われる。7:00 JST(22:00 UTC)の終値で買い、9:55 JST(00:55 UTC)の終値で売る。
合成の相場にはこの癖を入れていないので、ここで勝てば偶然。実データは日足しかないので確かめられない。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy

NS_MIN = 60_000_000_000


def is_gotobi(jst_ns: np.ndarray) -> np.ndarray:
    """JST の日付が五十日か(5・10・15・20・25 日と月末。土日なら前の金曜)。"""
    d = jst_ns.astype("datetime64[ns]").astype("datetime64[D]")
    out = np.zeros(len(d), dtype=bool)
    for shift in range(3):                      # 本来の五十日が土日なら、前の金曜に繰り上がる
        cand = d + np.timedelta64(shift, "D")
        dom = (cand - cand.astype("datetime64[M]")).astype(int) + 1
        last = (cand.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
        target = np.isin(dom, (5, 10, 15, 20, 25)) | (cand == last)
        dow = ((cand.astype(np.int64) + 3) % 7)            # 月曜 0
        if shift == 0:
            out |= target & (dow < 5)
        else:
            # d が金曜で、shift 日後(土日)が五十日
            d_dow = (d.astype(np.int64) + 3) % 7
            out |= target & (dow >= 5) & (d_dow == 4)
    return out


@STRATEGIES.register("gotobi", version="1", description="五十日の朝に買い、仲値(9:55 JST)で売る")
class Gotobi(Strategy):
    name = "gotobi"

    @dataclass
    class Params:
        timeframe: int = 5
        entry_jst: str = "07:00"
        exit_jst: str = "09:55"
        sl_pips: float = 30.0

    def prepare(self, bars, rng):
        def minutes(s):
            hh, mm = s.split(":")
            return int(hh) * 60 + int(mm)

        tc = bars.t + self.p.timeframe * NS_MIN                 # 足の終わり(終値の時刻)
        jst = tc + 9 * 60 * NS_MIN
        mod = (jst // NS_MIN) % 1440
        g = is_gotobi(jst)
        self.enter = g & (mod == minutes(self.p.entry_jst))
        self.exit = mod == minutes(self.p.exit_jst)

    def on_bar(self, i, ctx):
        if ctx.side != 0:
            if self.exit[i]:
                ctx.close()
            return
        if self.enter[i]:
            ctx.buy(sl=self.p.sl_pips * ctx.pip)
