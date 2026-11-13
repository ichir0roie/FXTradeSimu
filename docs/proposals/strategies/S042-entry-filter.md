# S042 entry_filter: 荒い局面・一方向の局面では新規を出さない包み

- 種類: 戦略(部品) / 状態: 実装済み(2026-11-11)
- 実装: `fxsim/strategies/builtin/entry_filter.py`(登録名 `entry_filter` v1)
- 親: E033(負けるタイミング)。評価は E035

## 仕組み
別の戦略(inner)を包み、新規(buy / sell / add)を出す足の状態が条件に掛かったら新規だけを止める。
止めた足でドテンしようとしたときは、反対の建玉の決済だけを行う(エンジンの取引時間外と同じ扱い)。決済・損切りは止めない。

| 引数 | 既定 | 意味 |
|---|---|---|
| inner / inner_params | ma_cross / {} | 包む戦略 |
| max_vol_ratio | 0(見ない) | ATR(vol_n) / 直前 vol_window 本の ATR の中央値がこれを越えたら止める |
| vol_n / vol_window | 14 / 500 | |
| max_er | 0(見ない) | er_n 本の効率比がこれを越えたら止める |
| er_n | 20 | |

時間帯・曜日で止めるのは、動作ケースの schedule(`fxsim/backtest/schedule.py`、tz: JST、hours、days、outside)を使う。
方向でたらめ版と組み合わせるときは `inner: direction_null, inner_params: {inner: <戦略>}`。

条件を全部 0 にすると、中の戦略と 1 円も違わない(E035 で確認)。
