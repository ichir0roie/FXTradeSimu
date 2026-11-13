"""MT5 の銘柄と口座の仕様、定数(MQL5 と同じ名前・値)。

銘柄(SymbolSpec)は「1 ロット = contract_size 通貨」「価格の最小単位 point」「数量の刻み」「損切り・利確を置ける最小距離
(stops_level、ポイント)」「スワップ」「手数料」を持つ。口座(AccountSpec)は「両建て(hedging)か、ネット(netting)か」
「レバレッジ」「マージンコール・ストップアウトの水準(証拠金維持率 %)」「マイナス残高を消すか(ゼロカット)」を持つ。

既定値は BrokerConfig から引く(同じブローカーなら 1 ペアのエンジンと同じ費用になるように):
    レバレッジ           BrokerConfig.leverage
    ストップアウト       BrokerConfig.losscut_level x 100(%)
    手数料(1 ロット片道) BrokerConfig.commission_per_10k x contract_size / 10,000
    スワップ             BrokerConfig.swap_mode = rates なら金利差 − 上乗せ(年率)、fixed なら 1 万通貨あたりの円を換算
    ゼロカット           BrokerConfig.zero_cut
MT5 にしか無い項目(口座の型・ロットの刻み・stops_level・マージンコール・両建ての証拠金・実行の型)は
BrokerConfig.mt5(辞書)で上書きする。値の出どころと注意は docs/MT5.md。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from ..broker import BrokerConfig

# ---- 定数(MQL5 と同じ値) ----------------------------------------------------
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT = 2, 3
ORDER_TYPE_BUY_STOP, ORDER_TYPE_SELL_STOP = 4, 5
ORDER_TYPE_BUY_STOP_LIMIT, ORDER_TYPE_SELL_STOP_LIMIT = 6, 7
ORDER_TYPE_NAMES = {0: "buy", 1: "sell", 2: "buy_limit", 3: "sell_limit", 4: "buy_stop", 5: "sell_stop",
                    6: "buy_stop_limit", 7: "sell_stop_limit"}

ORDER_TIME_GTC, ORDER_TIME_DAY, ORDER_TIME_SPECIFIED, ORDER_TIME_SPECIFIED_DAY = 0, 1, 2, 3

POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1

DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY = 0, 1, 2, 3
DEAL_REASON_CLIENT, DEAL_REASON_EXPERT, DEAL_REASON_SL, DEAL_REASON_TP, DEAL_REASON_SO = 0, 3, 4, 5, 6

TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_PRICE = 10015
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_MARKET_CLOSED = 10018
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_INVALID_EXPIRATION = 10022
TRADE_RETCODE_NO_CHANGES = 10025
TRADE_RETCODE_FROZEN = 10029
TRADE_RETCODE_POSITION_CLOSED = 10036
TRADE_RETCODE_LIMIT_POSITIONS = 10040
RETCODE_NAMES = {10004: "requote", 10006: "reject", 10008: "placed", 10009: "done", 10013: "invalid",
                 10014: "invalid_volume", 10015: "invalid_price", 10016: "invalid_stops", 10018: "market_closed",
                 10019: "no_money", 10022: "invalid_expiration", 10025: "no_changes", 10029: "frozen",
                 10036: "position_closed",
                 10040: "limit_positions"}

ACCOUNT_MARGIN_MODE_RETAIL_NETTING, ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 0, 2


@dataclass
class Mt5Options:
    """BrokerConfig.mt5 で上書きできる、MT5 にしか無い項目。"""
    account_mode: str = "hedging"         # hedging(両建て可、建玉ごとにチケット)/ netting(銘柄ごとに 1 つ)
    contract_size: float = 100_000.0      # 1 ロットの通貨量
    digits: int = 3                       # 価格の桁(USDJPY は 3 桁 = point 0.001)
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    stops_level: int = 0                  # 損切り・利確・待機注文を現在値から何ポイント以上離すか(0 = 制限なし)
    freeze_level: int = 0                 # 現在値からこのポイント以内に来た待機注文・損切り・利確は変更・削除できない
    margin_call_level: float = 100.0      # 証拠金維持率(%)がこれを割ったら通知(記録だけ。取引は止めない)
    hedged_margin: str = "larger_leg"     # 両建ての証拠金: larger_leg(多い方の数量だけ)/ full(両方)/ rate
    margin_hedged_rate: float = 0.5       # hedged_margin = rate のとき、両建て分の数量に掛ける割合
    execution: str = "market"             # market(滑って約定)/ instant(deviation を超えたら約定しない = リクオート)
    deviation_points: int = 20            # instant のときの既定の許容幅(ポイント)
    swap_points_long: float | None = None   # 指定すれば、スワップを「1 ロット 1 日あたりのポイント」で固定する
    swap_points_short: float | None = None
    max_positions: int = 200              # 口座に持てる建玉と待機注文の数の上限(超えたら 10040)
    # 足の中の判定(テスターのモデル)。MT5 の公式の説明(docs/MT5.md):
    #   every_tick  経路(1 分)の点 + 点の間で触れる確率。逆指値・損切り・利確は触れた点の気配で約定(窓なら滑る)、
    #               指値は指値の価格
    #   m1_points   経路の点だけ(点の間では触れない)。約定は every_tick と同じ
    #   m1_ohlc     MT5 の「1 分足 OHLC」: 経路の点で判定し、待機注文・損切り・利確は注文の価格で約定する
    #               (窓でも滑らない)
    #   open_prices MT5 の「始値のみ」: 足の始値・高値・安値・終値の 4 点で判定し、注文の価格で約定する
    tester_model: str = "every_tick"
    stop_out: str = "largest_first"       # largest_first(損失の大きい建玉から 1 つずつ、水準を戻すまで。実際の業者)
    #                                     # / all(全部を一度に。MT5 のテスター)
    # 指定すれば BrokerConfig の値より優先する(業者のひな形 MT5_PRESETS で使う)
    leverage: float | None = None
    stop_out_level: float | None = None   # %
    zero_cut: bool | None = None

    def __post_init__(self):
        if self.account_mode not in ("hedging", "netting"):
            raise ValueError("mt5.account_mode must be hedging or netting")
        if self.hedged_margin not in ("larger_leg", "full", "rate"):
            raise ValueError("mt5.hedged_margin must be larger_leg, full or rate")
        if self.execution not in ("market", "instant"):
            raise ValueError("mt5.execution must be market or instant")
        if self.tester_model not in ("every_tick", "m1_points", "m1_ohlc", "open_prices"):
            raise ValueError("mt5.tester_model must be every_tick, m1_points, m1_ohlc or open_prices")
        if self.stop_out not in ("largest_first", "all"):
            raise ValueError("mt5.stop_out must be largest_first or all")

    @property
    def order_price_fills(self) -> bool:
        """待機注文・損切り・利確を、窓でも注文の価格で約定させるか(MT5 のテスターの 1 分足 OHLC・始値のみ)。"""
        return self.tester_model in ("m1_ohlc", "open_prices")

    @classmethod
    def from_dict(cls, d: dict | None) -> Mt5Options:
        d = dict(d or {})
        preset = d.pop("preset", None)
        if preset is not None:
            if preset not in MT5_PRESETS:
                raise KeyError(f"unknown mt5 preset '{preset}'. known: {sorted(MT5_PRESETS)}")
            d = {**MT5_PRESETS[preset], **d}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown mt5 options {sorted(unknown)} (known: {sorted(known)})")
        return cls(**d)


# 業者のひな形(BrokerConfig.mt5 = {"preset": ...})。値は 2026-11 に業者の公開ページで確かめたもの(docs/MT5.md)。
# 費用(スプレッドの上乗せ・滑り・スワップの上乗せ)は BrokerConfig の側(fair / malicious など)で決める
MT5_PRESETS: dict[str, dict] = {
    # 国内の店頭(OANDA 証券の MT5 スタンダードを参考): 1 ロット 10 万通貨、最小 0.01・刻み 0.01、25 倍、
    # 証拠金維持率 100% 以下で損失の大きい建玉から 100% を超えるまで決済、マージンコールなし、手数料なし、
    # マイナス残高は客の負担(国内はゼロカットなし)。両建ての証拠金は多い方
    "jp_domestic": dict(account_mode="hedging", contract_size=100_000.0, volume_min=0.01, volume_step=0.01,
                        volume_max=10.0, margin_call_level=100.0, leverage=25.0, stop_out_level=100.0,
                        stop_out="largest_first", zero_cut=False, hedged_margin="larger_leg"),
    # 海外の高レバレッジ口座の典型: 500 倍、マージンコール 50%、ストップアウト 20%、ゼロカットあり
    "offshore": dict(account_mode="hedging", contract_size=100_000.0, volume_min=0.01, volume_step=0.01,
                     volume_max=100.0, margin_call_level=50.0, leverage=500.0, stop_out_level=20.0,
                     stop_out="largest_first", zero_cut=True, hedged_margin="larger_leg"),
    # MT5 のテスターそのもの(業者の値は jp_domestic と同じ。ストップアウトは全部を一度に、約定は注文の価格)
    "tester_ohlc": dict(account_mode="hedging", contract_size=100_000.0, volume_min=0.01, volume_step=0.01,
                        volume_max=10.0, margin_call_level=100.0, leverage=25.0, stop_out_level=100.0,
                        stop_out="all", zero_cut=False, tester_model="m1_ohlc"),
}


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    pip: float
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level: int
    freeze_level: int
    commission_per_lot: float             # 片道、口座通貨(円)
    swap_points_long: float | None
    swap_points_short: float | None

    def normalize_volume(self, volume: float) -> float:
        """刻みに切り捨て、上限で抑える。最小に満たなければ 0。"""
        steps = int(volume / self.volume_step + 1e-9)
        v = round(steps * self.volume_step, 8)
        v = min(v, self.volume_max)
        return v if v >= self.volume_min - 1e-12 else 0.0

    def valid_volume(self, volume: float) -> bool:
        if volume < self.volume_min - 1e-12 or volume > self.volume_max + 1e-12:
            return False
        k = volume / self.volume_step
        return abs(k - round(k)) < 1e-6

    def units(self, volume: float) -> float:
        return volume * self.contract_size

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AccountSpec:
    mode: str                 # hedging | netting
    currency: str
    leverage: float
    margin_call_level: float  # %
    stop_out_level: float     # %
    zero_cut: bool
    stop_out: str
    hedged_margin: str
    margin_hedged_rate: float
    max_positions: int

    def to_dict(self) -> dict:
        return asdict(self)


def specs(broker: BrokerConfig, pip: float, symbol: str = "USDJPY") -> tuple[SymbolSpec, AccountSpec, Mt5Options]:
    """BrokerConfig(と BrokerConfig.mt5)から、銘柄と口座の仕様を作る。"""
    o = Mt5Options.from_dict(broker.mt5)
    point = round(10.0 ** -o.digits, 10)
    sym = SymbolSpec(symbol, o.digits, point, pip, o.contract_size, o.volume_min, o.volume_step, o.volume_max,
                     o.stops_level, o.freeze_level, broker.commission_per_10k * o.contract_size / 10_000,
                     o.swap_points_long, o.swap_points_short)
    acct = AccountSpec(o.account_mode, "JPY", o.leverage if o.leverage is not None else broker.leverage,
                       o.margin_call_level,
                       o.stop_out_level if o.stop_out_level is not None else broker.losscut_level * 100.0,
                       o.zero_cut if o.zero_cut is not None else broker.zero_cut, o.stop_out, o.hedged_margin,
                       o.margin_hedged_rate, o.max_positions)
    return sym, acct, o
