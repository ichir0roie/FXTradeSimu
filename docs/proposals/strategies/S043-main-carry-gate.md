# S043 main_carry_gate: 本命のスワップ狙いに新規の関門を足したもの

- 種類: 戦略(本命の派生) / 状態: 実装済み(2026-11-12)。関門を足した B055 は不採用(E038)
- 実装: `fxsim/strategies/main/carry_gate.py`(登録名 `main_carry_gate` v1)
- 親: S040(main_carry_breakout)。評価は E038

## 仕組み
main_carry_breakout を継承し、`allowed(pair, i)`(新規を出してよいか)と、新規の数を数える ctx の包みだけを足す。
手仕舞い・口座の床・急変用の逆指値・量は main_carry_breakout のまま。carry_breakout.py には手を入れていない。

| 引数 | 既定 | 意味 |
|---|---|---|
| entry_vol_max | 0(見ない) | ATR(atr_n) / 直前 vol_window 本の ATR の中央値がこれを超えるペアには新規を出さない |
| vol_window | 500 | |
| counter_z | 0(見ない) | 60 本の対数リターン / (20 本のボラ x √60) を建てる向きで見て −counter_z を下回るなら新規を出さない |
| max_held | 0(見ない) | 口座が持つペアの数(この足で建てる分を含む)の上限 |

関門を全部 0 にすると、main_carry_breakout と 1 円も違わない(`tests/test_carry_gate.py`)。
