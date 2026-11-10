# 動作ケースとテストケース

戦略の案を「同じ手順で、機械的に」評価するための定義。

- **動作ケース**(`cases/behaviors/B###-*.yaml`): どう取引するか。トレード手法・判断に使う足・判断の間隔・取引時間・ブローカー・数量
- **テストケース**(`cases/tests/T###-*.yaml`): ある動作ケースを、どの相場パターンのリストで、どれだけの期間テストするか

```bash
python -m fxsim cases list              # 定義と最新の結果
python -m fxsim cases run --all         # 全テストケースを実行(定義とコードが前回と同じなら飛ばす)
python -m fxsim cases run T002 T003     # 指定したものだけ
python -m fxsim cases run T002 --force  # 変わっていなくても回し直す
python -m fxsim cases show T002         # 最新結果(パターンごとの損益と統計)
```
画面では `/cases`。

## 動作ケース

```yaml
id: B002
name: ブレイクアウト(H1・ロンドン〜NY 時間だけ・時間外は手仕舞い)
strategy: donchian_breakout          # 戦略の登録名(python -m fxsim list)
params: {entry_n: 48, exit_n: 24, atr_n: 24, sl_atr: 2.0}
timeframe: 60                        # 判断に使う足(分)。5 / 15 / 60 / 240 / 1440 など
schedule:
  decision_every: 1                  # 何本ごとに判断するか(H1 で 4 なら 4 時間ごと)
  hours: [["07:00", "21:00"]]        # 新規注文を出してよい時間帯。複数可。省略で終日。日またぎ ["22:00", "02:00"] も可
  tz: UTC                            # hours / days の基準: UTC / JST / LON / NY(固定の時差。夏時間は扱わない)
  days: [mon, tue, wed, thu, fri]    # 新規注文を出してよい曜日
  outside: close                     # 時間外の持ち高: hold(持ち越す)/ close(足の終値で手仕舞い、次の足の始値で約定)
broker: malicious                    # fair / malicious、または {preset: malicious, overrides: {hunt_prob: 0.5}}
engine: {units: 10000}               # 数量。{sizing: risk, risk_fraction: 0.01} で損切り幅からの資金管理
verify: {block_days: 1500}           # 任意。fxsim verify の既定(years / seeds / block_days)。省略で 30 年・1-10・250 日
```

- 判断は足の終値の時刻で行い、注文は次の足の始値で約定する
- 時間外でも、決済・損切り・利確は有効。時間外に反対方向のシグナルが出たら、新規は建てずに決済だけする

## テストケース

```yaml
id: T001
name: 日足 EMA 交差を、実データ 10 年と、その範囲のランダム化・完全ランダムで比べる
behavior: B004
period: {start: "2016-11-01", end: "2026-10-30"}   # または {years: 5}(実データ系は末尾から N 年)
patterns:
  - {market: real}                         # 実データそのもの(シードによらず 1 本)
  - {market: real_bridge, seeds: "1-10"}   # 実相場の日足を通り、足の中だけランダム
  - {market: real_bootstrap, seeds: "1-10"}# 実相場のリターンを並べ替えた別の相場
  - {market: malicious, seeds: "1-5"}      # 完全なランダム相場
  - {market: malicious, label: mal_strong, params: {stop_hunt: {prob: 0.12}}, seeds: [1]}   # パラメータ違い
detail_seeds: [1]                          # ローソク足と取引まで保存するシード(画面のチャート用)
```

- 期間を `start` / `end` で指定すると、実データ系はその期間を切り出し、合成相場は開始日を合わせる
- 同じ市場・パラメータ・シード・期間のパターンは、テストケースをまたいで同じ値動きになる(保存も共有される)。
  つまり、動作ケースだけを変えたテストケースどうしは、完全に同じ相場で比べられる

## 記録

- 実行ごとに `runs`(kind = test_case、proposal = テストケース ID)ができ、パターン = `paths`、バックテスト = `jobs`
- `case_defs` に定義のスナップショットとハッシュ、`case_runs` に「どの定義の版を、どのコードの版で、どの run で回したか」と
  損益の要約(パターンごとと総合の 平均・中央値・合計・標準偏差・最小/最大・プラスの割合・t 値)
- 定義を書き換えると画面に「要再実行」が出る

## 検証(ひとつのモデルが長期的に利益を出すか)

```bash
python -m fxsim verify B007 --years 30 --seeds 1-10 --block-days 250
python -m fxsim cases show V-B007           # 判定と表
```
画面は `/cases/V-B007`(判定、パターンの組ごとの損益、資産曲線の束、年ごとの損益)。

- 標準のパターンの組(実データ・並べ替え・実データ + 日中ランダム・偽薬・完全ランダム・ストレス)で、
  モデルと「方向だけでたらめ版」(direction_null)を対にして回し、C1〜C4 の条件で合否を判定する(`fxsim/verification.py`)
- `--block-days` は並べ替えのブロックの長さ。**戦略が見る期間より長く**する(短いと、戦略が使う構造を壊して不公平になる)
- `--dataset` で実データの CSV を差し替えられる(手元の 1 分足など)
- テストケースの YAML に `verify: true` を書いても同じ判定が付く(パターンの組は自分で選ぶ)
- 判定は 合格(C1〜C4 すべて)/ 条件付き(C1 と C2)/ 不合格。根拠の数字はすべて case_runs.summary_json に残る
