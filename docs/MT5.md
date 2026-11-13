# MT5 のテスターを真似たバックテスト(本命の戦略用)

本命の戦略(`fxsim/strategies/main/`)は、MetaTrader 5 の EA に移すことを前提に作る。
そのために、MT5 の口座と注文の決まりをシミュレーションに入れた(`fxsim/backtest/mt5/`)。
相場(`markets/`)とブローカーの振る舞い(`BrokerConfig` の fair / ecn / malicious)は、ほかの戦略と同じものを使う。

実口座で戦略を常駐させる仕組み(`fxsim trade`、`fxsim/trading/`)は別にある。手順は `docs/TRADING.md`。

## 戦略の区分

| 区分 | 場所 | エンジン | 中身 |
|---|---|---|---|
| 基本検証(`tier = "basic"`) | `strategies/builtin/` | 1 ペアのエンジン / 複数ペアのエンジン | 手法の一覧。E011 で全部不合格。物差しと市場モデルの検査に使い続ける |
| 本命(`tier = "main"`) | `strategies/main/` | MT5 のテスター(`engine = "mt5"`) | MT5 の EA に移す前提。共通の対策(`strategies/guards.py`)を入れて作る |

`python -m fxsim list` で区分ごとに出る。検証(`verify`)・実験・動作ケース・先読みの検査は、区分に関係なく同じ物差しで回る
(`build_backtester` がエンジンを選ぶ)。

## EA の書き方

```python
from fxsim.strategies.main.base import MainStrategy

@STRATEGIES.register("main_xxx")
class MainXxx(MainStrategy):
    name = "main_xxx"

    def signals(self, bars, rng):
        # 足 i の終値までで決まる値だけで(因果的に)3 つの列を作る
        return target, stop_dist, expected_move   # 向き(+1/0/−1)、損切りの幅(価格)、見込みの値幅(価格)
```

売買の決まり(終値の損切り・手仕舞い・新規のゲート・量)は `MainStrategy.on_bar` が共通で行う。
自分で注文を出したいときは `on_bar(i, ctx)` を書き、`ctx`(`backtest/mt5/context.py` の `Mt5Context`)を使う。

呼ばれる時点は「新しい足の最初のティック」。`on_bar(i, ctx)` の `i` は確定した足で、`ctx.bid` / `ctx.ask` は足 i+1 の始値の気配。
成行はその気配でその場で約定し、結果がすぐ返る。MQL5 なら `OnTick` で新しい足を判定し、確定した足(インデックス 1)を読む書き方に当たる。

| Python(`ctx`) | MQL5 |
|---|---|
| `ctx.buy(0.1, sl=.., tp=..)` / `ctx.sell(..)` | `CTrade::Buy` / `Sell` |
| `ctx.buy_limit` / `sell_limit` / `buy_stop` / `sell_stop` / `buy_stop_limit` / `sell_stop_limit` | `CTrade::BuyLimit` など |
| `ctx.position_close(ticket[, volume])` | `PositionClose` / `PositionClosePartial` |
| `ctx.position_close_by(ticket, by)` | `PositionCloseBy` |
| `ctx.position_modify(ticket, sl, tp)` | `PositionModify` |
| `ctx.order_modify` / `ctx.order_delete` | `OrderModify` / `OrderDelete` |
| `ctx.positions(magic)` / `ctx.orders(magic)` / `ctx.deals` | `PositionGet*` / `OrderGet*` / `HistoryDeal*` |
| `ctx.account()` | `AccountInfoDouble(ACCOUNT_BALANCE / EQUITY / MARGIN / MARGIN_FREE / MARGIN_LEVEL)` |
| `ctx.symbol()` | `SymbolInfo*`(`POINT`・`DIGITS`・`VOLUME_*`・`TRADE_STOPS_LEVEL`・スワップ) |
| 結果の `retcode` | `MqlTradeResult.retcode`(10009 約定、10008 待機注文、10016 損切りの位置が不正、10019 証拠金不足 など) |

数量はロット(1 ロット = `contract_size` 通貨)、損切り・利確・待機注文の価格は絶対値。

## シミュレーションしている MT5 の決まり

出どころ: MetaQuotes の公式の説明(2026-11 に確認)。確かめられなかったものは「未確認」と書く。

| 項目 | 決まり | 出どころ |
|---|---|---|
| 発動の側 | 買いの指値: Ask ≤ 価格、買いの逆指値: Ask ≥ 価格、売りの指値: Bid ≥ 価格、売りの逆指値: Bid ≤ 価格。ストップリミットは発動したら同じ側の指値になる。買いの建玉の損切り・利確は Bid、売りは Ask | [general_concept](https://www.metatrader5.com/en/terminal/help/trading/general_concept) |
| 約定の価格(ティックのモデル) | 指値は指値の価格(滑らない)。逆指値・損切り・利確は発動したティックの気配(窓なら滑る) | [testing_features](https://www.metatrader5.com/en/terminal/help/algotrading/testing_features) |
| 約定の価格(1 分足 OHLC・始値のみ) | 待機注文・損切り・利確は注文の価格で約定(窓でも滑らない)。始値のみでも、損切り・利確・待機注文は足の OHLC の点で発動する | 同上、[testing](https://www.mql5.com/en/docs/runtime/testing) |
| 止めの距離 | `SYMBOL_TRADE_STOPS_LEVEL`: 待機注文は買いを Ask、売りを Bid から、建玉の損切り・利確は買いを Bid、売りを Ask から測る。違反は 10016(待機注文の価格は 10015)。`FREEZE_LEVEL` の内側は変更できない(10029) | [articles/2555](https://www.mql5.com/en/articles/2555) |
| 証拠金 | FX: ロット x 契約サイズ / レバレッジ(円口座の USDJPY は x 価格)。両建ては多い方の数量(`hedged_margin = larger_leg`、ほかに full / rate) | [marketinfoconstants](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)、[margin_forex](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex) |
| ストップアウト | 実際の業者: 損失の一番大きい建玉から閉じ、水準を戻すまで繰り返す(`stop_out = largest_first`)。MT5 のテスター: 全部の建玉と注文を一度に閉じる(`all`)。マージンコールは状態だけ | [performing_deals](https://www.metatrader5.com/en/terminal/help/trading/performing_deals)、[testing](https://www.metatrader5.com/en/terminal/help/algotrading/testing) |
| ネットと両建て | ネット: 銘柄ごとに 1 つ。同じ向きは数量で加重平均、逆向きは減らし、多ければドテン(約定の entry = INOUT)。両建て: 建玉ごとにチケット、`close_by` で相殺(スプレッドは 1 回) | performing_deals |
| 実行の型 | market: 業者の価格で約定、deviation は使わない。instant: deviation を超えたらリクオート(10004) | performing_deals |
| 期限 | GTC / DAY(取引日の終わり)/ SPECIFIED / SPECIFIED_DAY | performing_deals |
| スワップ | ロールオーバー(サーバー時刻 0 時 = NY 17:00)に付く。水曜は 3 日分。金利差から計算する(`BrokerConfig.swap_mode = rates`)か、1 ロット 1 日のポイントで固定(`mt5.swap_points_long / short`) | [market_watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch) |
| レポートの指標 | 純益・総利益・総損失・プロフィットファクター・期待利得・リカバリーファクター・残高 / 有効証拠金のドローダウン(絶対・最大・相対)・AHPR・GHPR・Z スコア・LR 相関・連勝 / 連敗(`metrics["mt5"]`) | [testing_report](https://www.metatrader5.com/en/terminal/help/algotrading/testing_report) |

テスターのモデル(`BrokerConfig.mt5.tester_model`):

| モデル | 足の中で見る点 | 約定 | MT5 の対応 |
|---|---|---|---|
| `every_tick`(既定) | 1 分の経路の点 + 点の間で触れる確率(ブラウン橋) | ティックのモデルの決まり | 全ティック(現実に近い方に寄せている) |
| `m1_points` | 1 分の経路の点だけ | ティックのモデルの決まり | 1 分足 OHLC に近い点の数 |
| `m1_ohlc` | 1 分の経路の点だけ | 注文の価格(窓でも滑らない) | 1 分足 OHLC |
| `open_prices` | 足の始値・高値・安値・終値の 4 点 | 注文の価格 | 始値のみ |

MT5 のテスターと結果を合わせたいときは `m1_ohlc` / `open_prices` と `stop_out: all`(ひな形 `tester_ohlc`)。
判定(検証)には `every_tick` と実際の業者の決まり(`largest_first`)を使う。窓で滑らない約定は、損切りを使う戦略を良く見せる。

## 業者のひな形(`BrokerConfig.mt5 = {"preset": ...}`)

| ひな形 | 中身 | 出どころ |
|---|---|---|
| `jp_domestic` | 1 ロット 10 万通貨、0.01 ロット刻み、25 倍、証拠金維持率 100% 以下で損失の大きい建玉から 100% を超えるまで決済、マージンコールなし、手数料なし、ゼロカットなし(国内)、両建ての証拠金は多い方 | OANDA 証券の MT5 スタンダード([ロット](https://help.oanda.jp/oanda/faq/show/856)、[最小数量](https://www.oanda.jp/info/1232)、[ロスカット](https://www.oanda.jp/fx/ty3/about)) |
| `offshore` | 500 倍、マージンコール 50%、ストップアウト 20%、ゼロカットあり | 海外口座の典型(業者ごとに違う。Exness はストップアウト 0%、Pepperstone は 50%) |
| `tester_ohlc` | `jp_domestic` の値 + MT5 のテスターの約定(`m1_ohlc`)とストップアウト(`all`) | MT5 のテスターの説明 |

費用(スプレッドの上乗せ・滑り・スワップの上乗せ・ストップ狩り・ラストルック)は、ひな形ではなく `BrokerConfig` のプリセット
(fair / ecn / malicious)で決める。例: `{preset: malicious, overrides: {mt5: {preset: jp_domestic}}}`。

## 1 ペアのエンジンとの違いと、確かめたこと

- 成行だけの戦略なら、1 ペアのエンジンと 1 円も違わない(約定の価格・スプレッド・滑り・スワップ。`tests/test_mt5.py`)
- 違うところ:
  - 量: 証拠金が足りない注文は約定しない(10019)。1 ペアのエンジンは建てられる量まで勝手に減らす
  - 足の始まり: 始値のティック(窓)で起きる発動は、EA が呼ばれる前に処理する。窓で損切りに掛かった足で、すぐ新規を出せる
  - 建てた点より前: 建てた足の中では、建てた点より後だけで損切り・利確を見る。1 ペアのエンジンは足の頭から見ていて、
    建てる前の点の間で「触れた」ことにしうる(小さな差。1 ペアのエンジンの既知の近似)
- 足の中の点の時刻は、足の始まりから 1 歩ずつ(経路の無い足は足の始まり)。ドローダウンは足の終わりの値で測る
  (MT5 はティックごと。足の中の谷の分だけ小さく出る)
- 未対応: 複数銘柄の EA、`OnTimer`、ランダムな遅延、約定の部分約定(FOK / IOC)、証拠金の通貨の換算(円口座の USDJPY だけ)
