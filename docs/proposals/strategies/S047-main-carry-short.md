# S047 main_carry_short: 受け取る向きの信号で入り、数日で出る

- 種類: 戦略(対照) / 状態: 実装済み(2026-11-13)。採らない(E047)
- 実装: `fxsim/strategies/main/carry_hold.py`(登録名 `main_carry_short` v1)、テスト `tests/test_carry_hold.py`
- 親: S040(main_carry_breakout)。評価は E047(入った直後の伸びは E042・E043)

## 仕組み
| 引数 | 既定 | 意味 |
|---|---|---|
| signal | break | break = n_in 日の高値抜け / macd = MACD がシグナルを抜く / either = どちらか |
| hold / extend | 3 / true | 建ててから hold 本で出る。extend なら、持っている間に信号が点き直したら数え直す |
| min_carry / c2r_min | 0.02 / 0.4 | 下限は B049 と同じ |

## 結果と判定
E047 を参照。直後の伸びから費用を引いた残りは年 1,000〜2,000 円(10 万円・2,000 通貨)で、シャープ +0.14〜0.27。
持つ日数を延ばすほど B049 に近づく。採らない。
