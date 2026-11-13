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

## R014 real_pairs_bootstrap(2026-11-11 に追加、E026 で使用)
同じ 10 ペアの「別の歴史」。置く先の期間(`source.start` / `end`)の時刻の並びに、元の期間(`pool_start` / `pool_end`、
省くと DB の全部)の日付を `block_days` 本(既定 250)のブロックでつなぐ。全ペアで同じ日付の並び。

| 運ぶもの | 運ばないもの |
|---|---|
| 値動き(足の 4 点の形ごと。前の足の終値からの差で)、その日付の金利差(`Bars.rate_diff`)、ペアどうしの相関、ボラの塊 | スプレッド(置く先の期間の記録を使う。業者の費用は今の口座のもの) |

金利差を日付と一緒に運ぶので、値動きと金利のつながり(高金利通貨の値下がり、円高のときの金利差)が保たれる。
これまでの合成の相場(real_bootstrap・basket_bootstrap など)は金利差を本当の日付のまま当てたので、キャリーが有利に見えた
(ROADMAP の「合成の相場のスワップ」)。エンジン(1 ペア・複数ペア)と main_carry_breakout は、`Bars.rate_diff` があればそれを使う。

テスト: `tests/test_carry_trend.py`(金利差が全ペアで同じ日付の並びで運ばれる、スプレッドは置く先のまま、水準は置く先の最初の終値から)。

## R015 real_pairs_placebo とドリフト抜きの並べ替え(2026-11-11 に追加、E031 で使用)
`fxsim verify` で主要 10 ペアの戦略(`pair_market = "real_pairs"`)を判定するために足した。

| 市場 | 中身 | verify の組 |
|---|---|---|
| real_pairs_placebo(R015) | 同じ期間の real_pairs の 1 日ごとの値動き(足の 4 点の形ごと)に、全ペアで同じ日ごとの乱数の符号(±1)を掛ける。相関・ボラの塊・大きさは残る。金利差・スプレッドは本当の日付のまま(basket_placebo と同じ) | 偽薬(C3) |
| real_pairs_bootstrap の `source.demean: true` | ペアごとに元の期間(pool)の 1 本あたりの平均の値動きを引く。足の中の点は経過の割合だけ引く | ドリフト抜き(C5) |

テスト: `tests/test_carry_trend.py`(偽薬は大きさがそのままで全ペア同じ符号、ドリフト抜きは 1 本ごとに同じだけ引く、
verify の組と約定の時刻)。
