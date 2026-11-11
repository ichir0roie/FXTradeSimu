"""Wine の中の Windows 版 Python で動かす、MetaTrader5 への橋(rpyc の classic サーバー)。

    WINEPREFIX=~/.wine-mt5 wine 'C:\\Python311\\python.exe' fxsim/trading/wine_server.py [--port 18812]

fxsim の残りは import しない(Wine 側には rpyc・MetaTrader5・numpy<2 だけ入れる。docs/TRADING.md)。
classic サーバーは、つないだ相手に何でも実行させるので、127.0.0.1 だけで待ち受ける。
"""
import argparse
import logging

import MetaTrader5  # noqa: F401  起動の時点で入っているか確かめる
from rpyc.core import SlaveService
from rpyc.utils.server import ThreadedServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18812)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadedServer(SlaveService, hostname="127.0.0.1", port=args.port, reuse_addr=True,
                            protocol_config={"allow_public_attrs": True, "allow_pickle": True,
                                             "sync_request_timeout": 300})
    logging.info("MetaTrader5 bridge on 127.0.0.1:%d", args.port)
    server.start()


if __name__ == "__main__":
    main()
