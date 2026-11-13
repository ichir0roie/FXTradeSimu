"""常駐のループ: 接続 → 新しい足を待つ → 判断・注文 → … 。切れたらつなぎ直す。SIGTERM / SIGINT で止まる。"""
from __future__ import annotations

import logging
import signal
import threading
import time

from .config import TraderConfig, load_credentials
from .mt5 import MT5Client, MT5Error, load_module
from .trader import LiveTrader

log = logging.getLogger(__name__)


class Service:
    def __init__(self, cfg: TraderConfig, module=None, heartbeat_sec: float = 3600.0, runner=None):
        self.cfg = cfg
        self.module = module
        self.runner = runner            # runners.Mt5Runner(省けば設定から作る)
        self.heartbeat_sec = heartbeat_sec
        self.stop_event = threading.Event()
        self.client: MT5Client | None = None
        self.trader: LiveTrader | None = None

    def connect(self) -> LiveTrader:
        module = self.module if self.module is not None else load_module(self.cfg.mt5)
        self.client = MT5Client(module, self.cfg.mt5)
        creds = load_credentials(self.cfg.account)
        self.client.connect(creds)
        if self.runner is None:
            from ..runners.mt5 import Mt5Runner
            self.runner = Mt5Runner(self.cfg, module=self.module)
        # 1 銘柄は LiveTrader、複数ペア(symbols)は PortfolioTrader
        self.trader = self.runner.make_trader(self.client, expected_login=creds.login if creds else None)
        self.trader.startup()
        return self.trader

    def install_signals(self) -> None:
        def stop(signum, _frame):
            log.info("signal %s: stopping", signum)
            self.stop_event.set()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def run(self, once: bool = False) -> None:
        backoff = 5.0
        last_beat = 0.0
        while not self.stop_event.is_set():
            try:
                if self.trader is None or not self.client.alive():
                    if self.client is not None:
                        log.warning("terminal disconnected; reconnecting")
                        self.client.shutdown()
                    self.connect()
                    backoff = 5.0
                done = self.trader.poll()
                if once and done:
                    break
                if time.time() - last_beat >= self.heartbeat_sec:
                    a = self.client.account()
                    log.info("alive: equity %.2f %s, positions %d", a["equity"], a["currency"],
                             len(self.client.positions(self.cfg.symbol, self.cfg.magic)))
                    last_beat = time.time()
                self.stop_event.wait(self.trader.poll_interval)
            except (MT5Error, ConnectionError, EOFError, OSError) as e:
                log.error("MT5: %s (retry in %.0fs)", e, backoff)
                self.trader = None
                if once:
                    raise
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 300.0)
            except Exception as e:
                # 想定していない失敗でも常駐は止めない(記録して次の足で判断し直す)
                log.exception("unexpected error (retry in %.0fs)", backoff)
                if self.trader is not None:
                    self.trader.journal("error", message=repr(e))
                if once:
                    raise
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 300.0)
        if self.client is not None:
            self.client.shutdown()
        log.info("stopped")
