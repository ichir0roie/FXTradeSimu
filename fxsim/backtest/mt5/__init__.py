"""MT5 のストラテジーテスターを真似たバックテスト(本命の戦略 = EA を、MT5 に移す前提で検証する)。

    spec.py     銘柄・口座の仕様、MQL5 の定数、業者のひな形(MT5_PRESETS)
    orders.py   建玉・待機注文・約定・取引の結果
    tester.py   Mt5Tester(足の中の発動を起きた順に処理する。成績の形は 1 ペアのエンジンと同じ)
    context.py  EA から見える窓口(Mt5Context。CTrade・PositionGet* に寄せた名前)
    report.py   MT5 のテスターのレポートと同じ定義の指標
使い方と MT5 との対応は docs/MT5.md。
"""
from .context import AccountInfo, Mt5Context, OrderInfo, PositionInfo, SymbolInfo
from .orders import Mt5Deal, Mt5Order, Mt5Position, TradeResult
from .report import mt5_report
from .spec import MT5_PRESETS, AccountSpec, Mt5Options, SymbolSpec
from .tester import Mt5Tester

__all__ = ["MT5_PRESETS", "AccountInfo", "AccountSpec", "Mt5Context", "Mt5Deal", "Mt5Options", "Mt5Order",
           "Mt5Position", "Mt5Tester", "OrderInfo", "PositionInfo", "SymbolInfo", "SymbolSpec", "TradeResult",
           "mt5_report"]
