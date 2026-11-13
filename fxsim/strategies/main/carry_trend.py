"""S039 main_carry_trend: 本命の 2 本目。これまでの知見(E012〜E018)を 1 本にまとめた、日足のキャリー + トレンドの保険。

    向き      スワップを受け取る側だけ。金利差(基軸 − 決済)の符号の向きで、業者の受け取り側の取り分を引いた年率が
              min_carry 以上のときだけ持つ(E018: 払う向きに入らないと日足の戦略は全部良くなる。中身はキャリー)
    ゲート    終値が trend_n 日の移動平均と同じ側のときだけ持つ。キャリーの巻き戻し(急な円高)で手仕舞わせる保険
              (E018: 最悪のパターンを半分ほどにする。B008 と同じ形)
    損切り    ATR(atr_n) x stop_atr。終値で判定し(ひげでは切らない)、サーバーには非常用の逆指値だけを遠くに置く(guards)
    スプレッド 今のスプレッドが典型の 3 倍を超えていたら新規・信号の手仕舞いを出さない(E017、動作ケースの guard)
    量        損切りまでで有効証拠金の 0.5%。月初の有効証拠金の 9 割を床にする(guards)
    足        日足だけ(E012〜E017: 日中は費用を越えない)

金利差は、相場がペアを名乗るとき(real_pair の bars.pair)はそのペアの短期金利の差(8 通貨の月次)、それ以外は
USD − JPY の政策金利(rates_path)。月次の短期金利は月の平均なので、月の途中では少し先の値を含む(carry_trade と同じ近似)。
"""
from dataclasses import dataclass, field

import numpy as np

from ...data import load_rates
from ...data.rates import MULTI_RATES
from .. import STRATEGIES
from ..indicators import atr, sma
from .base import MainStrategy


@STRATEGIES.register("main_carry_trend", version="1",
                     description="本命: スワップを受け取る向きだけ + 200 日平均の保険 + 共通の対策(日足、MT5 前提)")
class MainCarryTrend(MainStrategy):
    name = "main_carry_trend"

    @dataclass
    class Params:
        timeframe: int = 1440
        guard: dict = field(default_factory=dict)
        trend_n: int = 200            # 0 ならゲートなし(純粋なキャリー)
        min_carry: float = 0.01       # 受け取る年率(業者の取り分を引いた後)の下限
        recv_markup: float = 0.0044   # 業者の受け取り側の取り分(OANDA、E017)
        atr_n: int = 20
        stop_atr: float = 3.0
        rates_path: str = "data/real/policy_rates_usd_jpy.csv"

    def signals(self, bars, rng):
        c = bars.close
        pair = getattr(bars, "pair", None)
        diff = (load_rates(MULTI_RATES).pair_differential(pair, bars.t) if pair
                else load_rates(self.p.rates_path).differential(bars.t))
        side = np.sign(diff).astype(int)
        carry = np.abs(diff) - self.p.recv_markup
        target = np.where(carry >= self.p.min_carry, side, 0)
        if self.p.trend_n > 0:
            ma = sma(c, self.p.trend_n)
            above = np.sign(np.where(np.isfinite(ma), c - ma, 0.0)).astype(int)
            agree = np.isfinite(ma) & (above == side)
            target = np.where(agree, target, 0)
        a = atr(bars.high, bars.low, c, self.p.atr_n)
        self.warmup = max(self.p.trend_n, self.p.atr_n)
        return target, self.p.stop_atr * a, a * np.sqrt(self.g.hold_days_est)
