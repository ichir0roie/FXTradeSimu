# S044 main_carry_ema: 本命のスワップ狙いの入りと出を EMA の向きに替えたもの

- 種類: 戦略(本命の派生) / 状態: 実装済み(2026-11-13)。動作ケース B059(候補)
- 実装: `fxsim/strategies/main/carry_ema.py`(登録名 `main_carry_ema` v1)、テスト `tests/test_carry_ema.py`
- 親: S040(main_carry_breakout)。評価は E044

## 仕組み
main_carry_breakout を継承し、入りの信号(`signal_in`)と出の信号(`signal_out`)だけを差し替える。この 2 つは
carry_breakout.py から切り出した(B049 の成績は切り出しの前後で 1 円も変わらない)。金利差の下限・キャリー・トゥ・リスク・
受け取る向きだけ・急変用の逆指値・口座の床・証拠金の上限・スプレッドの門は main_carry_breakout のまま。

| 引数 | 既定 | 意味 |
|---|---|---|
| ema_fast / ema_slow | 10 / 30 | 速い EMA が遅い EMA より受け取る向きの側にある間は入る。反対側に回ったら出る |
| entry | ema | break にすると、入りだけ n_in 日の高値抜け(S040 と同じ)にする |
| min_hold | 0 | 建ててからこの本数が経つまでは EMA では出ない |
| signal_every | 1 | EMA の入りと出を判定する間隔(5 = 週に 1 回)。金利差の下限・口座の床は毎日効く |

## 結果と判定
E044 を参照。B059(EMA5/20・週に 1 回判定)は verify 合格。シャープは B049 と区別が付かず、年率が高い分だけ最大 DD が大きい。
