"""ブローカー(約定のしかたと費用)。市場の悪意とは別に、取引相手としての悪意と費用をここで表す。

費用
- スプレッド: 市場モデルのスプレッドに、ブローカーの上乗せ(spread_markup_pips)と下限(spread_floor_pips)を掛ける
- 手数料: 片道ごとに 1 万通貨あたり commission_per_10k 円(ECN 口座)
- スワップ: swap_mode="rates" なら政策金利の差(data/real/policy_rates_usd_jpy.csv)から毎日計算し、
  ブローカーが買い・売りの両側から swap_markup(年率)を取る。水曜は 3 日分。
  "fixed" なら 1 日あたり固定額(1 万通貨あたり円)
- スリッページ: fair は上下対称、malicious は不利な方向だけ・スプレッドに比例

約定(intrabar)
- "path": 損切り・利確・ロスカットは、足の中の基準解像度の経路で最初に越えた点で判定する。
  越えた 1 歩が飛び(その足のふつうの 1 歩の gap_k 倍より大きい動き。窓・指標・急変の段・スプレッドの急拡大)なら、
  飛んだ先の価格で約定する(損切り価格より大きく不利になる)。ふつうの 1 歩なら、相場は点と点の間も連続して動くので
  損切り価格で約定する(1 分ごとの点で約定させると、刻みの分だけ毎回滑ってしまう)。
  損切りと利確が同じ足に入ったら、実際に先に届いた方
- "pessimistic": 足の高安だけで判定し、両方届いた足は損切りが先

取引相手としての悪意
- 有利な約定の横取り(positive_slippage=False): 利確水準を窓で飛び越えても利確価格で約定
- ストップ狩り(hunt_prob / hunt_pips): 逆指値の近くまで来た足で、確率で刺される
- ラストルック(last_look_prob): 判断した時点から客に有利に動いた新規注文を、確率で拒否する(次の足で約定し直す)
- ゼロカット(zero_cut): False なら、窓で証拠金を超えて負けた分(マイナス残高)を客に請求する
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

BROKER_PRESETS: dict[str, dict] = {
    # 良心的な国内の店頭口座: 市場のスプレッドだけ、手数料なし、スワップの上乗せは小さい
    "fair": dict(slippage_pips=0.1, adverse_only=False, slippage_spread_k=0.0, stop_slippage_mult=1.0,
                 positive_slippage=True, hunt_prob=0.0, hunt_pips=0.0, spread_markup_pips=0.0,
                 spread_floor_pips=0.0, commission_per_10k=0.0, swap_mode="rates", swap_markup=0.0025,
                 last_look_prob=0.0, zero_cut=True),
    # ECN 口座: 店頭より狭い生のスプレッド(上乗せがマイナス)+ 片道手数料(10 万通貨あたり約 3.5 ドル)
    "ecn": dict(slippage_pips=0.1, adverse_only=False, slippage_spread_k=0.2, stop_slippage_mult=1.5,
                positive_slippage=True, hunt_prob=0.0, hunt_pips=0.0, spread_markup_pips=-0.2,
                spread_floor_pips=0.05, commission_per_10k=50.0, swap_mode="rates", swap_markup=0.0025,
                last_look_prob=0.0, zero_cut=True),
    # 取引相手として悪意のあるブローカー
    "malicious": dict(slippage_pips=0.2, adverse_only=True, slippage_spread_k=0.5, stop_slippage_mult=2.5,
                      positive_slippage=False, hunt_prob=0.35, hunt_pips=3.0, spread_markup_pips=0.3,
                      spread_floor_pips=0.5, commission_per_10k=0.0, swap_mode="rates", swap_markup=0.01,
                      last_look_prob=0.3, zero_cut=False),
}


@dataclass
class BrokerConfig:
    name: str = "custom"
    slippage_pips: float = 0.1        # スリッページの標準的な大きさ(pip)
    adverse_only: bool = False        # True なら不利な方向だけ(|N|)
    slippage_spread_k: float = 0.0    # その足の平均スプレッド(pip)にこの係数を掛けた分を加える
    stop_slippage_mult: float = 1.0   # 逆指値(損切り)の約定はこの倍だけ滑る
    positive_slippage: bool = True    # 利確が窓で有利に飛んだとき、その分を客に渡すか
    hunt_prob: float = 0.0            # 逆指値まで hunt_pips 以内に来た足で刺される確率
    hunt_pips: float = 0.0
    spread_markup_pips: float = 0.0   # 市場のスプレッドへの上乗せ(pip。ECN のように狭くするならマイナス)
    spread_floor_pips: float = 0.0    # スプレッドの下限(pip)
    commission_per_10k: float = 0.0   # 片道の手数料(円 / 1 万通貨)
    swap_mode: str = "fixed"          # fixed | rates
    swap_long_per_10k: float = 0.0    # fixed のとき: 円 / 日 / 1 万通貨
    swap_short_per_10k: float = 0.0
    swap_markup: float = 0.0          # rates のとき: ブローカーが両側から取る年率(0.01 = 1%)
    rates_path: str = "data/real/policy_rates_usd_jpy.csv"
    last_look_prob: float = 0.0       # 客に有利に動いた新規注文を拒否する確率
    zero_cut: bool = True             # マイナス残高を 0 に戻す(客に請求しない)か
    intrabar: str = "path"            # path | pessimistic
    gap_k: float = 6.0                # path: 1 歩の |動き| の中央値のこの倍(正規なら約 4σ)を超える 1 歩は飛び
    # path: 逆指値・指値に「触れた」の判定。bridge は点と点の間をブラウン橋とみなし、両端が手前にあっても
    # 間で触れる確率 exp(-2ab/σ²) で触れたことにする(最初に触れた瞬間に指値の価格で約定する、という
    # 連続の解釈と矛盾しない)。points は点で越えたときだけ判定する(E004〜E006 の版)。points で指値の価格で
    # 約定させると、越えたら指値の価格・戻ったら持ち続け、を後から選べることになり、損切りを使う戦略が
    # ランダムウォークでも年 60〜110 pip 得をしていた(R008)
    touch: str = "bridge"             # bridge | points
    leverage: float = 25.0
    losscut_level: float = 0.5        # 有効証拠金 / 必要証拠金
    # MT5 のテスター(backtest/mt5)にしか無い項目の上書き(口座の型・ロットの刻み・stops_level など)。
    # 1 ペアのエンジンと複数ペアのエンジンは使わない。項目は backtest/mt5/spec.py の Mt5Options
    mt5: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.swap_mode not in ("fixed", "rates"):
            raise ValueError("swap_mode must be fixed or rates")
        if self.intrabar not in ("path", "pessimistic"):
            raise ValueError("intrabar must be path or pessimistic")
        if self.touch not in ("bridge", "points"):
            raise ValueError("touch must be bridge or points")

    @classmethod
    def preset(cls, name: str, overrides: dict | None = None) -> BrokerConfig:
        if name not in BROKER_PRESETS:
            raise KeyError(f"unknown broker preset '{name}'. known: {sorted(BROKER_PRESETS)}")
        return cls(name=name, **{**BROKER_PRESETS[name], **(overrides or {})})

    def to_dict(self) -> dict:
        return asdict(self)
