# S046 main_carry_hold: 金利差の順位の上位を、受け取る向きで常に持つ

- 種類: 戦略(対照) / 状態: 実装済み(2026-11-13)。採らない(E047)
- 実装: `fxsim/strategies/main/carry_hold.py`(登録名 `main_carry_hold` v1)、テスト `tests/test_carry_hold.py`
- 親: S040(main_carry_breakout)、S041(main_carry_rank)。評価は E047

## 仕組み
| 引数 | 既定 | 意味 |
|---|---|---|
| top_n / rank_by | 5 / carry | 受け取る向きの金利差(carry)かキャリー・トゥ・リスク(c2r)の高い順に上位を持つ。毎日決め直す |
| min_carry | 0 | 順位に入れる金利差の下限 |
| lt_n | 250 | 終値がこの日数前より受け取る向きに動いているペアだけ新規を出す(0 = 見ない) |
| exit / stop_atr | profit_or_stop / 3 | 順位から外れたら、合計がプラスか ATR x stop_atr 逆行で閉じる(now = すぐ閉じる) |

## 結果と判定
E047 を参照。スワップは B049 の 2 倍取るが、最大 DD も 2〜3 倍で、シャープは B049 より下。採らない。
