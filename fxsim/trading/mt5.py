"""MT5 の端末とのやり取り。

MetaTrader5 の Python パッケージは Windows でしか動かない。Linux では Wine の中で MT5 と Windows 版の Python を動かし、
rpyc でつなぐ(backend: rpyc。Wine 側は wine_server.py)。Windows なら直接 import する(backend: metatrader5)。
どちらも同じ関数の集まり(initialize / copy_rates_from_pos / order_send ...)なので、MT5Client はその上に薄く乗る。
テストでは同じ関数を持つ偽物(tests/test_trading.py の FakeMT5)を渡す。
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .config import Credentials, MT5Config
from .feed import TIMEFRAMES

log = logging.getLogger(__name__)

# MetaTrader5 の定数(モジュールに無いときの既定値。値は MQL5 のドキュメントと同じ)
_CONSTANTS = {
    "ORDER_TYPE_BUY": 0, "ORDER_TYPE_SELL": 1,
    "TRADE_ACTION_DEAL": 1, "TRADE_ACTION_SLTP": 6,
    "ORDER_TIME_GTC": 0,
    "ORDER_FILLING_FOK": 0, "ORDER_FILLING_IOC": 1, "ORDER_FILLING_RETURN": 2,
    "TRADE_RETCODE_DONE": 10009, "TRADE_RETCODE_PLACED": 10008, "TRADE_RETCODE_NO_CHANGES": 10025,
    "POSITION_TYPE_BUY": 0, "POSITION_TYPE_SELL": 1,
    "ACCOUNT_TRADE_MODE_DEMO": 0,
    "TIMEFRAME_M1": 1, "TIMEFRAME_M5": 5, "TIMEFRAME_M15": 15, "TIMEFRAME_M30": 30,
    "TIMEFRAME_H1": 16385, "TIMEFRAME_H4": 16388, "TIMEFRAME_D1": 16408,
}
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2      # symbol_info.filling_mode のビット


class MT5Error(RuntimeError):
    pass


@dataclass
class SymbolSpec:
    name: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    filling_mode: int
    stops_level: int          # 損切り・利確を置ける現在値からの最小距離(ポイント)


@dataclass
class LivePosition:
    ticket: int
    side: int                 # +1 買い / -1 売り
    volume: float             # ロット
    price_open: float
    sl: float | None
    tp: float | None
    time: int                 # サーバー時刻(秒)


@dataclass
class OrderResult:
    ok: bool
    retcode: int
    comment: str
    order: int = 0
    deal: int = 0
    price: float = 0.0
    volume: float = 0.0
    request: dict | None = None


def _local(x):
    """rpyc 越しのオブジェクトを手元に持ってくる。手元のものはそのまま返す。"""
    if type(x).__module__.startswith("rpyc"):
        import rpyc
        return rpyc.classic.obtain(x)
    return x


_REMOTE_HELPER = """
import pickle as _pickle
import MetaTrader5 as _mt5

def _fxsim_plain(x):
    if hasattr(x, "_asdict"):
        return {"__nt__": True, **{k: _fxsim_plain(v) for k, v in x._asdict().items()}}
    if isinstance(x, tuple):
        return tuple(_fxsim_plain(e) for e in x)
    return x

def _fxsim_call(name, payload):
    args, kwargs = _pickle.loads(payload)
    attr = getattr(_mt5, name)
    r = attr(*args, **kwargs) if callable(attr) else attr
    return _pickle.dumps(_fxsim_plain(r))
"""


def _from_plain(x):
    if isinstance(x, dict) and x.get("__nt__"):
        return SimpleNamespace(**{k: _from_plain(v) for k, v in x.items() if k != "__nt__"})
    if isinstance(x, tuple):
        return tuple(_from_plain(e) for e in x)
    return x


class RemoteMT5:
    """Wine の中の MetaTrader5(wine_server.py)を、手元のモジュールのように呼ぶ。

    引数と戻り値は pickle で丸ごと受け渡す(rpyc の参照のまま渡すと、MetaTrader5 の C の関数が dict と認めない)。
    名前付きタプル(AccountInfo など)は手元に MetaTrader5 が無くても読めるよう SimpleNamespace にする。
    相手は 127.0.0.1 の自分のサーバーだけ(pickle を読むので、ほかの相手につながない)。
    """

    def __init__(self, host: str = "localhost", port: int = 18812, timeout_sec: float = 300.0):
        import rpyc
        self._conn = rpyc.classic.connect(host, port, keepalive=True)
        self._conn._config["sync_request_timeout"] = timeout_sec
        self._conn.execute(_REMOTE_HELPER)
        self._call = self._conn.namespace["_fxsim_call"]

    def close(self) -> None:
        self._conn.close()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name.isupper():                                   # 定数
            return _from_plain(pickle.loads(self._call(name, pickle.dumps(((), {})))))

        def call(*args, **kwargs):
            return _from_plain(pickle.loads(self._call(name, pickle.dumps((args, kwargs)))))
        return call


def load_module(cfg: MT5Config):
    if cfg.backend == "rpyc":
        return RemoteMT5(cfg.host, cfg.port, cfg.timeout_ms / 1000 * 5)
    import MetaTrader5 as mt5  # backend: metatrader5(Windows だけ)
    return mt5


class MT5Client:
    def __init__(self, module, cfg: MT5Config | None = None, sleep=time.sleep):
        self.m = module
        self.cfg = cfg or MT5Config()
        self._specs: dict[str, SymbolSpec] = {}
        self.sleep = sleep

    def c(self, name: str) -> int:
        """MetaTrader5 の定数。"""
        v = getattr(self.m, name, None)
        return int(v) if v is not None else _CONSTANTS[name]

    # ---- 接続 ----
    def connect(self, credentials: Credentials | None = None) -> None:
        """端末を起動・接続する。credentials があればその口座にログインする(無ければ端末がいまログインしている口座)。"""
        kw = {"timeout": self.cfg.timeout_ms}
        if self.cfg.path:
            kw["path"] = self.cfg.path
        if credentials is not None:
            kw.update(login=credentials.login, password=credentials.password, server=credentials.server)
        if not self.m.initialize(**kw):
            raise MT5Error(f"initialize failed: {self.last_error()}")
        self._specs.clear()
        # 端末を起動した直後は、ログインが済むまで口座の番号が 0 のまま返る
        for _ in range(30):
            a = self.m.account_info()
            if a is not None and int(_local(a).login) != 0:
                break
            self.sleep(1.0)

    def shutdown(self) -> None:
        try:
            self.m.shutdown()
        except Exception:      # 切れているときの shutdown は失敗してよい
            log.debug("shutdown failed", exc_info=True)

    def last_error(self):
        try:
            return _local(self.m.last_error())
        except Exception:
            return None

    def alive(self) -> bool:
        try:
            return self.m.terminal_info() is not None and self.m.account_info() is not None
        except Exception:
            return False

    # ---- 参照 ----
    def account(self) -> dict:
        a = self.m.account_info()
        if a is None:
            raise MT5Error(f"account_info failed: {self.last_error()}")
        a = _local(a)
        return {k: getattr(a, k) for k in ("login", "server", "currency", "balance", "equity", "margin",
                                            "margin_free", "leverage", "trade_mode", "margin_mode", "company")}

    def is_demo(self) -> bool:
        return int(self.account()["trade_mode"]) == self.c("ACCOUNT_TRADE_MODE_DEMO")

    def account_kind(self) -> str:
        """demo | real(コンテストの口座は real として扱う。デモでないものは全部本番とみなす)。"""
        return "demo" if self.is_demo() else "real"

    def trade_allowed(self) -> bool:
        t = self.m.terminal_info()
        return bool(t is not None and getattr(t, "trade_allowed", False))

    def symbol(self, name: str) -> SymbolSpec:
        if name in self._specs:
            return self._specs[name]
        if not self.m.symbol_select(name, True):
            raise MT5Error(f"symbol_select({name}) failed: {self.last_error()}")
        s = self.m.symbol_info(name)
        if s is None:
            raise MT5Error(f"symbol_info({name}) failed: {self.last_error()}")
        s = _local(s)
        spec = SymbolSpec(name, int(s.digits), float(s.point), float(s.trade_contract_size), float(s.volume_min),
                          float(s.volume_max), float(s.volume_step), int(s.filling_mode),
                          int(getattr(s, "trade_stops_level", 0)))
        self._specs[name] = spec
        return spec

    def tick(self, name: str):
        t = self.m.symbol_info_tick(name)
        if t is None:
            raise MT5Error(f"symbol_info_tick({name}) failed: {self.last_error()}")
        return _local(t)

    def timeframe(self, minutes: int) -> int:
        if minutes not in TIMEFRAMES:
            raise ValueError(f"MT5 has no timeframe of {minutes} minutes (supported: {sorted(TIMEFRAMES)})")
        return self.c(f"TIMEFRAME_{TIMEFRAMES[minutes]}")

    def rates(self, name: str, minutes: int, count: int) -> np.ndarray:
        """最新(作りかけの足を含む)から遡って count 本。古い順。

        ログインの直後や、端末がまだ持っていない古い足を頼んだときは、履歴が届くまで失敗する(-1 Call failed)
        ので、少し待ってやり直す。
        """
        tf = self.timeframe(minutes)
        for k in range(6):
            r = self.m.copy_rates_from_pos(name, tf, 0, int(count))
            if r is not None and len(r) > 0:
                return np.asarray(_local(r))
            if k < 5:
                self.sleep(2.0)
        raise MT5Error(f"copy_rates_from_pos({name}) failed: {self.last_error()}")

    def server_offset(self, name: str, tolerance_sec: float = 120.0) -> int | None:
        """サーバー時刻 − UTC(秒、30 分単位)。推定できないときは None。

        気配が古い(市場が閉じている)ときと、ログイン直後で気配の時刻がまだ 0 のときは推定できない。
        """
        tick = self.tick(name)
        diff = float(tick.time) - time.time()
        offset = round(diff / 1800.0) * 1800
        if abs(offset) > 14 * 3600 or abs(diff - offset) > tolerance_sec:
            return None
        return int(offset)

    def positions(self, name: str, magic: int) -> list[LivePosition]:
        ps = self.m.positions_get(symbol=name)
        if ps is None:
            raise MT5Error(f"positions_get({name}) failed: {self.last_error()}")
        out = []
        buy = self.c("POSITION_TYPE_BUY")
        for p in _local(ps):
            if int(p.magic) != int(magic):
                continue
            out.append(LivePosition(int(p.ticket), 1 if int(p.type) == buy else -1, float(p.volume),
                                    float(p.price_open), float(p.sl) or None, float(p.tp) or None, int(p.time)))
        return out

    # ---- 注文 ----
    def _filling(self, spec: SymbolSpec) -> int:
        if spec.filling_mode & SYMBOL_FILLING_FOK:
            return self.c("ORDER_FILLING_FOK")
        if spec.filling_mode & SYMBOL_FILLING_IOC:
            return self.c("ORDER_FILLING_IOC")
        return self.c("ORDER_FILLING_RETURN")

    def _send(self, req: dict) -> OrderResult:
        r = self.m.order_send(req)
        if r is None:
            return OrderResult(False, -1, f"order_send returned None: {self.last_error()}", request=req)
        r = _local(r)
        code = int(r.retcode)
        ok = code in (self.c("TRADE_RETCODE_DONE"), self.c("TRADE_RETCODE_PLACED"))
        return OrderResult(ok, code, str(getattr(r, "comment", "")), int(getattr(r, "order", 0)),
                           int(getattr(r, "deal", 0)), float(getattr(r, "price", 0.0)),
                           float(getattr(r, "volume", 0.0)), req)

    def market(self, name: str, side: int, volume: float, *, sl: float | None = None, tp: float | None = None,
               position: int | None = None, magic: int, deviation: int, comment: str) -> OrderResult:
        """成行。position を渡すとその建玉の決済(volume は決済する量)。"""
        spec = self.symbol(name)
        tick = self.tick(name)
        req = {
            "action": self.c("TRADE_ACTION_DEAL"),
            "symbol": name,
            "volume": float(volume),
            "type": self.c("ORDER_TYPE_BUY") if side > 0 else self.c("ORDER_TYPE_SELL"),
            "price": float(tick.ask if side > 0 else tick.bid),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment[:31],
            "type_time": self.c("ORDER_TIME_GTC"),
            "type_filling": self._filling(spec),
        }
        if sl:
            req["sl"] = round(float(sl), spec.digits)
        if tp:
            req["tp"] = round(float(tp), spec.digits)
        if position is not None:
            req["position"] = int(position)
        return self._send(req)

    def modify(self, name: str, ticket: int, sl: float | None, tp: float | None, *, magic: int) -> OrderResult:
        spec = self.symbol(name)
        req = {"action": self.c("TRADE_ACTION_SLTP"), "symbol": name, "position": int(ticket), "magic": int(magic),
               "sl": round(float(sl), spec.digits) if sl else 0.0,
               "tp": round(float(tp), spec.digits) if tp else 0.0}
        r = self._send(req)
        if r.retcode == self.c("TRADE_RETCODE_NO_CHANGES"):
            r.ok = True
        return r
