# S045 main_trend_free: スワップの向きを見ずに(または払う向きだけで)信号の向きに持つ対照

- 種類: 戦略(対照) / 状態: 実装済み(2026-11-13)。採らない(E046)
- 実装: `fxsim/strategies/main/trend_free.py`(登録名 `main_trend_free` v1)、テスト `tests/test_trend_free.py`
- 親: S040(main_carry_breakout)。評価は E046

## 仕組み
main_carry_breakout を継承し、向きの決め方(`_want`)と入り・出の信号だけを替える。

| 引数 | 既定 | 意味 |
|---|---|---|
| sides | both | recv = 受け取る向きだけ / both = 金利差を見ない / pay = 払う向きだけ |
| signal | break | break = n_in 日の高値抜けで入り n_out 日のチャネルで出る / ema = EMA の向き / mom = mom_n 日前との比較 |
| ema_fast / ema_slow / mom_n / signal_every | 5 / 20 / 63 / 1 | |

sides = recv・signal = break は main_carry_breakout と 1 円も違わない。both・pay はエンジンの swap_side_only を外して回す。

## 結果と判定
E046 を参照。両方向・払う向きは 5 つの信号すべてで負け、10 万円の口座では証拠金不足になる。採らない。
