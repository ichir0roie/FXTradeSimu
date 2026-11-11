# R013 real_pairs: 相場の DB の主要ペアを日付をそろえて 1 つの口座で回す

- 種類: 市場モデル / 状態: 実装済み(E025 で使用)
- 実装: `fxsim/markets/models/real_pairs.py`(登録名 `real_pairs` v1)、`PortfolioBacktester` のペアごとの pip(`pips`)
- 親: R012(real_pair)、basket_*(S032〜S036)

## 設計
ペアごとに real_pair(円に換算、記録のスプレッド、MT5 のテスターと同じ足の中の順番、ペアの金利差)を作り、全ペアにある足だけを残す。
先頭のペアの MarketData の meta["legs"] に全ペア。複数ペアの戦略(Strategy.multi)は PortfolioBacktester で、証拠金と有効証拠金を共有して回る。
円に換算した pip はペアごとに違うので、PortfolioBacktester に `pips` を渡す(basket_* は全ペア 0.01 で変わらない)。

| 項目 | 既定 |
|---|---|
| ペア | USDJPY・EURUSD・GBPUSD・AUDUSD・NZDUSD・USDCHF・USDCAD・EURJPY・GBPJPY・AUDJPY |
| 足 | `source.timeframe`(既定 D1) |
| 期間 | `source.start` / `end` |
| 日足の約定のスプレッド | `source.daily_fill_hour`(R012 と同じ) |

## 測り方
`tests/test_pair_eval.py`: どれかのペアに無い日は全ペアから落ちる、ペアごとの pip が PortfolioBacktester に渡る。

## 限界
- 円への換算は期間の最初の値で固定(R012 と同じ)
- 口座全体のロスカットは足の終値で見る(日足では足の中の急変を見ない)
