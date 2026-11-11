# CLAUDE.md — FXTradeSimu で案を出し、評価し、磨き続けるための手順

このリポジトリは「悪意ある為替相場」を合成し、FX 戦略を長期間シミュレーションする場所。
Claude の役目は、市場モデルと戦略の **案を複数出し → 実装し → 同じ物差しで評価し → 記録し → 次の案へ** を回すこと。
Claude 上ではシミュレーションだけを行い、結果は概要(`docs/results/<run_key>/`、CSV と JSON)だけをコミットする。
DB(`data/fxsim.sqlite`)は手元だけのもので、コミットしない。
画面(`web/`、Next.js)は利用者のローカル専用。

## 置き場所

| 何を | どこに |
|---|---|
| 市場モデルの部品(悪意の種類) | `fxsim/markets/components/*.py`(1 クラス = 1 つの仕組み) |
| 市場モデルの案 | `fxsim/markets/models/<案>.py`(1 ファイル = 1 案、`@MARKETS.register`) |
| 戦略の案(基本検証) | `fxsim/strategies/builtin/<案>.py`(1 ファイル = 1 案、`@STRATEGIES.register`)。手法の一覧と物差しの確認用 |
| 本命の戦略の案(MT5 前提) | `fxsim/strategies/main/<案>.py`。`MainStrategy` を継承して `signals` を書く。共通の対策は `fxsim/strategies/guards.py`、MT5 の決まりは `fxsim/backtest/mt5/`(`docs/MT5.md`) |
| 複数ペアの戦略の案 | 同じ場所。`fxsim/strategies/basket.py` の `BasketStrategy` / `CrossSectional` を継承(`multi = True`)。相場は `basket_*` |
| 実データ | `data/real/`(README に出典とライセンス)。対円 7 ペアと 8 通貨の金利は `python -m fxsim fetch-fred` で取り直せる |
| MT5 から溜めた足(コミットしない) | 相場の DB `data/market.sqlite`(時間足ごとのテーブル、時刻は UTC)。`python -m fxsim fetch-mt5 configs/market_data/*.yaml` で書き足し、`python -m fxsim market-data` で一覧。実データのパスに `db:USDJPY/M1` と書けば読める。1 分足は MT5 から 3 か月ほどしか遡れないので、取り直せない。消さない |
| 実験の設定 | `configs/*.yaml` |
| 動作ケース / テストケース | `cases/behaviors/*.yaml` / `cases/tests/*.yaml`(書式は `docs/CASES.md`) |
| 実験の結果(コミットする) | 概要 `docs/results/<run_key>/`(run.json・paths.csv・jobs.csv。書式は `docs/results/README.md`) |
| 実験の結果(コミットしない) | DB `data/fxsim.sqlite`(資産曲線・取引・ローソク足まで全部)。手元だけ。要るときは同じ設定で回し直す |
| 案の記録(仮説・設計・結果・判定・次の一手) | `docs/proposals/{markets,strategies,experiments}/<ID>-<名前>.md`(雛形 `docs/proposals/TEMPLATE.md`) |
| MT5 の口座での常駐(fxsim trade) | コード `fxsim/trading/`、設定 `configs/trading/*.yaml`、状態と記録 `data/trading/<name>/`(コミットしない)、手順 `docs/TRADING.md`。動かすのは利用者の端末だけ |
| 全体の設計 | `docs/ARCHITECTURE.md`(パッケージ構成と依存の向き)、悪意の一覧 `docs/MARKET_MALICE.md` |
| これまでの評価の整理 | 投資の知見 `docs/INSIGHTS.md`、開発項目 `docs/ROADMAP.md`、戦略の候補 `docs/CANDIDATES.md` |

ID の付け方: 市場モデル `M###`(合成)/ `R###`(実データ系)、戦略 `S###`、実験 `E###`、動作ケース `B###`、テストケース `T###`。
既存の最大番号 + 1 を使う(`ls docs/proposals/*/`)。

## 1 周の回し方

1. **現状を読む**
   - `python -m fxsim leaderboard` と `python -m fxsim leaderboard --kind markets`
   - 直近の提案書の「次の一手」(`docs/proposals/`)と、`docs/ROADMAP.md`・`docs/CANDIDATES.md`
2. **案を複数出す**(1 回に 2〜4 案)。それぞれ「何を、なぜ、どう測れば当否が分かるか」を提案書に先に書く
3. **実装する**
   - 市場の悪意を増やす → 部品クラスを書き、既存モデルを複製した新しいモデル(新 ID・version)に組み込む。
     既存モデルの既定値を書き換えるときは version を上げ、提案書に旧値を残す
   - 戦略 → `Strategy` を継承。`prepare` は因果的な numpy 計算だけ、`on_bar(i)` は足 i の終値までを見る。
     量を変える戦略は `ctx.buy(units=)`、ナンピン・積み増しは `ctx.add`、前回の結果は `ctx.trades`。
     賭け方(マーチンゲールなど)は `fxsim/strategies/staking.py` を使う
4. **検査する**: `python -m pytest -q` と `python -m fxsim check-lookahead`
5. **評価する**(下の「物差し」)。結果は自動で SQLite に入る
6. **記録する**: 提案書に run_key・主要な数字(表を `python -m fxsim report <run_key>` から貼る)・判定・次の一手。
   DB の番号(run #N)は手元の DB ごとに変わるので、提案書には run_key を書く。
   提案書の表に無い切り口は、回し直さずに概要から取る(`fxsim.reports.read_summary("<run_key>")`)
7. **コミット**: コードと概要(`docs/results/`)と提案書を一緒に。既定の DB への実験は終わったら自動で概要が
   書き出される(demo は除く)。DB(`*.sqlite`)はコミットしない

## 物差し(同じ条件で比べること)

### 市場モデルの案
```bash
python -m fxsim eval-market <model> --years 3 --seeds 1-6 --proposal M00X
python -m fxsim ablate <model> --years 3 --seeds 1-4 --proposal M00X    # どの部品が効いているか
```
- **現実らしさ**(`realism_score`、0–100): `fxsim/markets/evaluation.py` の TARGETS の範囲に入るか。80 未満なら原因を直す
- **悪用されにくさ**: パネル戦略(random_entry 以外。1 時間足の 5 本 + rsi_reversion + 5 分足の scalping)の
  **素のシャープ(gross_sharpe、コスト抜き)の絶対値の最大**。
  悪意ある市場の目標は「どの単純な規則にも安定した素の優位性が無い」= 0 に近いほど良い(目安 < 0.3)。
  正にも負にも大きいなら、その規則(または逆張り)で稼げる穴がある
- **コストと悪意の重さ**: random_entry の net と、fair / ecn / malicious ブローカーの差。
  どの費用が効いたかは report の「費用の内訳」(スプレッド・スリッページ・手数料・スワップ、狩り・拒否の回数)
- **判定はパターン単位の t 値**(report / 画面の `t値` 列。平均 ÷ (標準偏差 / √パターン数))。|t| ≥ 2 で有意とみなす。
  取引単位の素のシャープは、同じ経路の中の取引が独立でないので有意さを大きく見せる(E001 の注意を参照)
- 穴は短い期間では見えないことがある(M003 は 3 年 × 6 では素のシャープ 0.63、10 年 × 8 ではブレイクアウトが 8 本中 7 本プラス)。
  採用の前に 10 年 × 8 パターン以上で確かめる

### 戦略の案
```bash
python -m fxsim eval-strategy <name> --params '{...}' --years 5 --seeds 1-10 --null --patterns --proposal S00X
```
- 実データ(`real`、日足のみなので日足の戦略だけ)と、ランダム相場の複数パターン(`real_bridge` / `real_bootstrap` /
  `real_bridge_malicious`)で回る。パターンごとの損益と、平均・中央値・合計(総合)・標準偏差・最小/最大・プラスの割合が出る
- 判定は **パターン単位の t 値・中央値・プラスの割合**、次に **malicious ブローカーでの平均**。実データ 1 本だけ良い戦略は過剰適合を疑う
- 市場側の穴(M003 の順張り、R002/R004/M002 の逆張り)に乗っただけの勝ちを、戦略の力と取り違えない。
  穴の一覧は各提案書の「次の一手」
- random_entry(`--null`)より悪いなら、その戦略はコスト以下の情報しか持っていない

### 動作ケースとテストケース(機械的な評価)
「どう取引するか」を `cases/behaviors/B###-*.yaml`、「どの相場パターンで、どれだけの期間試すか」を
`cases/tests/T###-*.yaml` に書き、`python -m fxsim cases run --all` で回す。定義とコードが前回と同じなら飛ばすので、
いつでも全部流してよい。書式は `docs/CASES.md`。戦略の案を比べるときは、ここに足すのが一番手間が少ない。

### スワップの向き
- 全部の動作ケースは `engine: {swap_side_only: true}`(スワップを払う向きの新規は出さない、E022)。新しい動作ケースにも書く
- 判定では、方向でたらめ版にも同じ決まりが掛かる。決まりで上がった分(キャリー)は戦略の手柄ではない

### 実データだけでの評価(主要 10 ペア x 期間)
```bash
python -m fxsim pair-eval <動作ケース>...    # または --all。相場の DB の 10 ペアを期間に切り、方向でたらめ版と対で回す(E016)
python -m fxsim pair-eval --all --broker oanda  # 仮想の悪意を外した OANDA の口座の条件(E017)。日足は --daily-fill-hour 2 も
python -m fxsim pair-eval --all --broker oanda --daily-fill-hour 2 --swap-side  # スワップを受け取る向きの新規だけ(E022)
```
- 合成の相場を使わない評価。判定(R1〜R5)は `fxsim/pair_eval.py` の先頭。相場は `real_pair`(R012)
- 1 本の歴史(USDJPY)で合格しても、ペアを変えると消えることがある(B045)。verify と両方で見る

### ひとつのモデルの長期検証(合否)
```bash
python -m fxsim verify <動作ケース>...      # 既定は 30 年・1-10・ブロック 250 日。動作ケースの verify で上書き
python -m fxsim verify --all               # 全部の動作ケース(定義とコードが前回と同じなら飛ばす)
```
- 「このモデルは長期的に利益を出すか」はこれで判定する(E002)。1 本のバックテストの結果で判断しない
- 有意性は並べ替え(real_bootstrap)で測る。real_bridge は日足の戦略では実データのコピーなので数に入れない
- 合格の条件と落とし穴は `fxsim/verification.py` の先頭と `docs/proposals/experiments/E002-model-verification.md`
- 判定の仕組みを変えたら、陽性対照のテスト(`test_verification_passes_a_model_with_a_real_edge`)が合格のままか確かめる
- 並べ替えのブロックは戦略が見る期間より長くする(動作ケースの `verify: {block_days: ...}`)
- 日中の戦略は日足の実データでは回らないので、C1 の「実データでプラス」を外して並べ替えだけで判定する(2026-11 の判断。
  日中の実データが手に入れば戻す)
- C5(値動きの優位性): ドリフトを抜いた並べ替えで、スワップを除いた損益が C1 と同じ基準を満たすこと。円安とスワップだけで勝つ
  買いに偏った戦略を落とす(E011)。キャリーそのものを狙う戦略は C5 を通らないので、金利と値動きがつながった実データで別に評価する
- 判定の基準だけを変えたときは、回し直さずに `python -m fxsim verify --all --rejudge` で記録済みの結果に当て直す(概要も書き直す)

### 長期の基準実験
`configs/long_run_v2.yaml`(E003、市場モデル v2)。大きな変更の後は回し直し、提案書の表と比べる。
`configs/long_run_baseline.yaml` は E001(v1 の市場モデルの時点)の設定として残している。

## コードの約束

- 乱数は必ず市場の部品なら `ctx.rng`、戦略なら `prepare(bars, rng)` の rng から。グローバル乱数を使わない(再現性)
- 同じ値が並びうる並べ替えは `kind="stable"`。既定のクイックソートは同点の順番が CPU の SIMD で変わり、
  同じシードでも端末ごとに別の相場になっていた(E007、`test_markets_do_not_depend_on_the_cpu_simd_features`)
- 悪意のずれを数十分以上かけた直線で入れない。動き出しを見てから乗れば残りが取れる(M006)
- 1 つの部品は `GenContext` の配列を読み書きするだけ。部品どうしの受け渡しは `ctx.meta`(例: `crowd_mix`, `regime`)
- 価格を持続的にずらす悪意は `apply_block_increments`、一時的なヒゲは `ctx.logp` に直接足す
- 実データの骨格に合わせた後(ANCHOR 段)に値動きを足す部品は、骨格の終値(`ctx.meta["anchor"]["idx"]`)を
  動かさないこと(`FlashCrashOverlay._limits` を参照)
- 尖度は数本の急変で大きく動くので、現実らしさの採点は切り詰めた尖度(`kurt_h1_trim`)で行う
- 新しい部品は `ctx.event(idx, kind, ...)` でイベントを残す(画面のマーカーと集計に出る)。
  画面に色を付けるなら `web/components/Charts.tsx` の `EVENT_STYLE` にも足す
- pandas 3 は配列を読み取り専用で返すことがある。書き換える配列は `to_numpy(copy=True)`
- SQL は `fxsim/storage/`(writer.py / queries.py)と `web/lib/queries/` にだけ書く。場所(パス)は `fxsim/paths.py` から取る
- コミット前に `ruff check fxsim tests` と `python -m pytest -q`。画面を触ったら `cd web && npx next build`
- 結果に影響しないはずの整理をしたら、同じ設定を回して成績が 1 円も変わらないことを確かめる(結果は乱数のシードで決まる)
- ワーカー数の既定は論理 CPU の半分 − 2(手元の画面が固まらないように)。ワーカーは nice 10 で動く。
  増やすなら `FXSIM_WORKERS` か `--workers`、設定の `workers`(数は結果に影響しない)
- 指標・コミット・PR の本文は普通の文体で書く

## よく使うコマンド

```bash
python -m fxsim list                                   # 市場モデル・戦略・ブローカー
python -m fxsim run configs/quick.yaml                 # 動作確認(数十秒)
python -m fxsim report [RUN_ID] --patterns             # 結果を Markdown で
python -m fxsim demo --years 0.5 --delay-ms 200        # 画面で追うためのゆっくりしたシミュレーション
python -m fxsim summary <run_key>                      # 概要を docs/results/ に書き出し直す(--all で DB の全実行)
sqlite3 data/fxsim.sqlite "SELECT * FROM v_strategy_summary WHERE run_id = 1"
python -m fxsim compact <run_key> --detail-equity-minutes 43200   # 手元の DB の資産曲線を間引く(成績は変わらない)
python -m fxsim overfit <run_key> --market basket_bootstrap   # 戦略の集まりの PBO と Deflated Sharpe(試した数の補正)
```

## 画面(利用者のローカル)
```bash
cd web && npm install && npm run dev   # http://localhost:3000(FXSIM_DB で DB を差し替え)
npm run screenshot                     # 別の端末で。docs/screenshots/ に保存
```
