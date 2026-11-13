"""常駐の設定(configs/trading/*.yaml)。

例:
    name: demo-usdjpy-b001
    behavior: B001                 # cases/behaviors の動作ケース(戦略・パラメータ・足・取引時間・数量)
    symbol: USDJPY                 # MT5 の銘柄名(ブローカーによって USDJPY.m などの接尾辞が付く)
    magic: 20261110                # この常駐が出した注文の印。ほかの EA や手動の建玉には触らない
    account: demo                  # 口座の情報を .env.demo から読む(本番は real と .env.real、allow_real: true)
    mt5: {backend: rpyc, host: localhost, port: 18812}
口座の番号・パスワード・サーバーは設定に書かず、リポジトリ直下の .env.<account>(コミットしない)に置く。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from ..paths import TRADING_DIR, account_env_file, resolve

ACCOUNTS = ("demo", "real")

# シミュレーションに使うブローカー。実口座の約定は MT5 が決めるので、シミュレーションは素直にする
# (malicious の狩りや拒否を入れると、実口座では起きていない手仕舞いをシミュレーションがしてしまう)
DEFAULT_SIM_BROKER = {"preset": "fair", "overrides": {"slippage_pips": 0.0, "swap_mode": "fixed"}}


@dataclass
class MT5Config:
    backend: str = "rpyc"            # rpyc(Wine の中の wine_server.py へつなぐ)/ metatrader5(Windows で直接)
    host: str = "localhost"          # rpyc のとき
    port: int = 18812
    path: str | None = None          # 端末(terminal64.exe)の場所。省略で自動。1 つの端末は 1 つの口座にしかつながらない
    timeout_ms: int = 60_000

    def __post_init__(self):
        if self.backend not in ("rpyc", "metatrader5"):
            raise ValueError("mt5.backend must be rpyc or metatrader5")


@dataclass
class TraderConfig:
    name: str
    behavior: str
    symbol: str
    magic: int
    account: str = "demo"            # demo | real。口座の情報を .env.<account> から読み、つないだ口座の種別と照らす
    mt5: MT5Config = field(default_factory=MT5Config)
    volume_scale: float = 1.0        # 実口座の数量 = シミュレーションの数量(通貨) x これ
    max_lots: float = 1.0            # 1 回の注文と建玉の合計の上限(ロット)。超える注文は出さない
    history_bars: int = 3000         # 起点を決めるときに遡る本数(戦略の warmup より十分長く)
    max_history_bars: int = 60_000   # 起点からの本数の上限。超えたら起点を進める
    poll_seconds: str | float = "auto"   # 新しい足を見に行く間隔(秒)。auto は足の長さ / 120 を 5〜60 秒に収めたもの
                                         # (日足・4 時間足 60 秒、1 時間足 30 秒、5 分足 5 秒)。負荷を抑えるため
    protect: float = 0.9             # 口座の床 = 月初の有効証拠金 x これ。割ったらこの常駐の建玉を全部決済し、
                                     # その月は新規を出さない(0 = 床を置かない。同じ口座の常駐どうしで床を共有する)
    gap_mult: float = 1.5            # 最悪の損失 = 逆指値までの損失 x これ(窓で滑る分)
    floor_check_seconds: float = 60.0    # 足の確定を待つ間も、この間隔で有効証拠金を床と比べる
    deviation_points: int = 20       # 成行の許容スリッページ(ポイント)
    server_utc_offset_hours: str | float = "auto"   # MT5 のサーバー時刻 − UTC。auto は気配の時刻から推定
    pip: float | None = None         # 省略で銘柄の桁数から(3 桁・2 桁なら 0.01、5 桁・4 桁なら 0.0001)
    sim_broker: dict = field(default_factory=lambda: dict(DEFAULT_SIM_BROKER))
    dry_run: bool = False            # True なら注文を出さず、出すはずの注文を記録するだけ
    allow_real: bool = False         # account: real のとき、これも true でないと動かない
    comment: str = "fxsim"           # 注文のコメント(MT5 の画面に出る。31 文字まで)
    # 複数ペアの戦略(Strategy.multi、例: B047)だけ。MT5 の銘柄名の並び。symbol(足の確定と時刻の基準)も含め、
    # 対円でない決済通貨を円に直すため USDJPY と、USDCHF・USDCAD など決済通貨のペアも入れる
    symbols: list[str] | None = None
    # 日足の注文を出すサーバーの時(None = 足の確定ですぐ)。判断はサーバーの 0 時(ロールオーバー)だが、その 1 時間は
    # スプレッドが平常の数倍に広がる(E017)ので、2 なら 2 時まで待ってから出す。待っている注文は state.json に残す
    fill_server_hour: int | None = None
    # 動作ケースの戦略のパラメータへの上書き(試験用。例: 再生で取引の多い設定を流す)。本番では空にしておく
    strategy_params: dict = field(default_factory=dict)
    # 複数ペアだけ: 判断の足の金利差を、端末のスワップの点数(業者が実際に付ける分)から出す(False なら FRED の短期金利)。
    # OANDA の点数は「次のロールオーバーで付く分」で、週末の分が付く日は 3 日分になる(2026-11-11 に確かめた)。
    # 判断のたびに観測を残し、直近 10 回の中央値を 1 日分として使う。観測が broker_swap_min_obs 回に満たなければ FRED
    broker_swap: bool = True
    broker_swap_min_obs: int = 3
    # 複数ペアだけ: 新規を出す直前のスプレッドが、直近 24 時間の 1 分足のスプレッドの中央値のこの倍を超えていたら待つ
    # (0 = 見ない)。待つのは fill_wait_hours まで。過ぎたらそのまま出す
    fill_max_spread_mult: float = 3.0
    fill_wait_hours: float = 6.0

    def __post_init__(self):
        if isinstance(self.mt5, dict):
            self.mt5 = _typed(MT5Config, self.mt5)
        if self.account not in ACCOUNTS:
            raise ValueError(f"account must be one of {ACCOUNTS}")
        if self.account == "real" and not self.allow_real:
            raise ValueError("account: real needs allow_real: true")
        if self.magic <= 0:
            raise ValueError("magic must be a positive integer")
        if self.max_lots <= 0 or self.volume_scale <= 0:
            raise ValueError("max_lots and volume_scale must be positive")
        if isinstance(self.poll_seconds, str) and self.poll_seconds != "auto":
            raise ValueError("poll_seconds must be 'auto' or a number")
        if not 0 <= self.protect < 1:
            raise ValueError("protect must be in [0, 1)")
        if isinstance(self.server_utc_offset_hours, str) and self.server_utc_offset_hours != "auto":
            raise ValueError("server_utc_offset_hours must be 'auto' or a number")
        if self.symbols is not None:
            if isinstance(self.symbols, str):
                self.symbols = [self.symbols]
            if self.symbol not in self.symbols:
                raise ValueError("symbol (the lead symbol) must be one of symbols")
        if self.fill_server_hour is not None and not 0 <= int(self.fill_server_hour) <= 23:
            raise ValueError("fill_server_hour must be 0..23")

    @property
    def state_dir(self) -> Path:
        return TRADING_DIR / self.name

    def poll_interval(self, timeframe_minutes: int) -> float:
        if self.poll_seconds == "auto":
            return float(min(60.0, max(5.0, timeframe_minutes * 60 / 120)))
        return float(self.poll_seconds)

    def to_dict(self) -> dict:
        return asdict(self)


def _typed(cls, value: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    return cls(**value)


def load_trader_config(path: str | Path) -> TraderConfig:
    with open(resolve(path), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _typed(TraderConfig, data)


@dataclass
class Credentials:
    login: int
    password: str
    server: str

    def __repr__(self) -> str:          # ログや例外にパスワードを出さない
        return f"Credentials(login={self.login}, server={self.server!r})"


def load_credentials(account: str, path: Path | None = None) -> Credentials | None:
    """.env.<account> の MT5_LOGIN / MT5_PASSWORD / MT5_SERVER。
    ファイルが無ければ None(端末がいまログインしている口座を使う)。

    環境変数は見ない(シェルに残ったデモの値で本番の設定が動く、といった取り違えを避ける)。
    """
    path = path or account_env_file(account)
    if not path.exists():
        return None
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        env[k.strip()] = v
    missing = {"MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"} - set(env)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}")
    return Credentials(int(env["MT5_LOGIN"]), env["MT5_PASSWORD"], env["MT5_SERVER"])
