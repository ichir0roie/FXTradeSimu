"""ブローカー(約定のしかた)。市場の悪意とは別に、取引相手としての悪意をここで表す。

- スリッページ: fair は上下対称、malicious は不利な方向だけ・ボラに比例
- 有利な約定の横取り: positive_slippage=False なら、窓で利確水準を飛び越えても利確価格で約定
- ストップ狩り: あなたの逆指値まで hunt_pips 以内に来た足で、確率 hunt_prob で刺される
- スワップ: 買い/売りの 1 日あたりの受け払い(1 万通貨あたり、円)。水曜は 3 日分
- ロスカット: 有効証拠金 / 必要証拠金 が losscut_level を割ったら強制決済
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

BROKER_PRESETS: dict[str, dict] = {
    # コストの下限を見るための基準。スプレッドは市場モデルのものを払う。
    "fair": dict(slippage_pips=0.1, adverse_only=False, slippage_spread_k=0.0, stop_slippage_mult=1.0,
                 positive_slippage=True, hunt_prob=0.0, hunt_pips=0.0,
                 swap_long_per_10k=20.0, swap_short_per_10k=-40.0),
    # 取引相手として悪意のあるブローカー。
    "malicious": dict(slippage_pips=0.2, adverse_only=True, slippage_spread_k=0.5, stop_slippage_mult=2.5,
                      positive_slippage=False, hunt_prob=0.35, hunt_pips=3.0,
                      swap_long_per_10k=10.0, swap_short_per_10k=-90.0),
}


@dataclass
class BrokerConfig:
    name: str = "fair"
    slippage_pips: float = 0.1        # スリッページの標準的な大きさ(pip)
    adverse_only: bool = False        # True なら不利な方向だけ(|N|)
    slippage_spread_k: float = 0.0    # その足の平均スプレッド(pip)にこの係数を掛けた分を加える
    stop_slippage_mult: float = 1.0   # 逆指値(損切り)の約定はこの倍だけ滑る
    positive_slippage: bool = True    # 利確が窓で有利に飛んだとき、その分を客に渡すか
    hunt_prob: float = 0.0            # 逆指値まで hunt_pips 以内に来た足で刺される確率
    hunt_pips: float = 0.0
    swap_long_per_10k: float = 0.0    # 円/日/1 万通貨
    swap_short_per_10k: float = 0.0
    leverage: float = 25.0
    losscut_level: float = 0.5        # 有効証拠金 / 必要証拠金

    @classmethod
    def preset(cls, name: str, overrides: dict | None = None) -> "BrokerConfig":
        if name not in BROKER_PRESETS:
            raise KeyError(f"unknown broker preset '{name}'. known: {sorted(BROKER_PRESETS)}")
        return cls(name=name, **{**BROKER_PRESETS[name], **(overrides or {})})

    def to_dict(self) -> dict:
        return asdict(self)
