# 設計

```
configs/*.yaml ──> fxsim.experiments.runner ──> data/fxsim.sqlite <── web/(Next.js、ローカル専用)
                         │                             ▲
                         ├─ markets: 経路を作る ───────┤ paths / candles / market_events
                         └─ backtest: 売買を回す ──────┘ jobs / equity / trades(途中経過も逐次)
```

## パッケージ構成

依存の向きは上から下だけ(下の層は上の層を知らない)。SQL は `storage/`(Python)と `web/lib/queries/`(画面)にだけ書く。

```
fxsim/
  paths.py             リポジトリ内の決まった場所(DB・cases・実データ)。場所はここからだけ取る
  calendar.py          取引カレンダー(UTC、週は日曜 22:00〜金曜 22:00、ロールオーバー 22:00)
  numerics.py          AR(1) フィルタ、t 分布など
  registry.py          名前 -> ファクトリ(市場モデル・戦略で共用)
  data/                データ型
    market.py            MarketData(基準解像度の mid とスプレッド、時間足のキャッシュ)
    bars.py              Bars(任意の時間足、bid/ask の OHLC)と make_bars
    real.py              実データの読み込み(汎用 CSV / HistData)
    rates.py             金利の表(スワップの計算に使う。通貨ごとの列も持てる)
    fred.py              FRED から対円 7 ペアの日足と 8 通貨の短期金利を取る(fetch-fred)
  markets/             市場モデル
    base.py              Stage / Component / GenContext / MarketModel
    blocks.py            経路をブロック単位で扱う補助(POST_PATH の部品が使う)
    components/          部品(volatility, regime, shocks, malice, extreme, real, spread)
    models/              案(random_walk, stylized, malicious, real_based, basket(複数ペア))
    evaluation.py        現実らしさ(stylized facts)と悪意の指標
  backtest/            バックテスト
    broker.py            ブローカー(スプレッドの上乗せ、手数料、スワップ、スリッページ、約定の仕方、悪意)
    schedule.py          判断の間隔・取引時間
    engine.py            1 ポジションのバックテスト
    portfolio.py         複数ペアを 1 つの口座で回すバックテスト(ペアごとに engine の Leg、残高・証拠金は共有)。
                         build_backtester が戦略に合うエンジンを選ぶ
    mt5/                 MT5 のテスターを真似たバックテスト(本命の戦略用。docs/MT5.md)。spec(銘柄・口座・定数・業者のひな形)/
                         orders / tester(Mt5Tester)/ context(EA の窓口 Mt5Context)/ report(MT5 のレポートの指標)
    metrics.py           成績
    lookahead.py         先読み検査
  strategies/          戦略(base.py / basket.py(複数ペア)/ indicators.py / staking.py(賭け方))
    builtin/             基本検証の戦略(tier = basic、1 ファイル 1 戦略)
    main/                本命の戦略(tier = main、engine = mt5。base.MainStrategy を継承)
    guards.py            本命の戦略に共通の対策(量・費用の門・回転の上限・終値の損切り・ドローダウンの止め)
  catalog.py           組み込みの案をすべて登録する
  storage/             SQLite
    schema.py            テーブルとビュー
    db.py                接続・トランザクション・JSON
    writer.py            書き込み(計画の行、相場パターン、ジョブの途中経過と結果、ケースの記録)
    queries.py           読み出し(レポート・統計・ケース)
  experiments/         実験
    config.py            設定(ExperimentConfig と StorageConfig / LiveConfig、YAML の書式)
    runner.py            計画(plan)と実行(経路ごとに 1 プロセス)。PathTask / JobTask
    stats.py             パターンごとの統計(平均・中央値・合計・t 値など)
  cases/               動作ケースとテストケース
    model.py             BehaviorCase / TestCase / Pattern
    loader.py            YAML を読んで検査する
    runner.py            テストケースを実験に直して回し、記録する
  verification.py      ひとつのモデルの長期検証の判定(C1〜C4、パターンの組、年ごと、参考の Deflated Sharpe)
  overfit.py           試した数の補正(Deflated Sharpe Ratio、PBO)
  reports/             Markdown のレポート(run_report / leaderboard / cases_overview)
  cli.py               python -m fxsim ...
web/                   Next.js 16(App Router)+ better-sqlite3 + lightweight-charts
  lib/types.ts           API の型(サーバーと画面で共有)
  lib/queries/           SQL(runs / jobs / cases)
  lib/stats.ts           パターンごとの統計(Python の stats.py と同じ定義・同じキー)
  lib/useJobStream.ts    ジョブの差分を 1 秒ごとに取りに行くフック
  components/            画面(RunsView / RunView / JobView / CasesView / CaseView(検証)/ Charts / FanChart)
```

## 市場モデル = 部品の並び

`MarketModel.generate(years, seed)` は、段階(Stage)の順に部品を呼ぶ。部品は共有の `GenContext` の配列を書き換える。

| 段階 | 役割 | 部品 |
|---|---|---|
| SOURCE | 時刻列と基準を外から与える | RealReplay, RealSkeleton |
| VOLATILITY | ボラの倍率、予定された出来事 | SessionSeasonality, StochasticVolatility, NewsShocks |
| DRIFT | ドリフト | RegimeSwitching |
| INNOVATION | ショックの分布 | FatTails |
| RETURNS | ジャンプ | WeekendGap |
| (経路を組み立てる) | `logp = log(p0) + cumsum(drift + vol * shock + jump)` | |
| POST_PATH | 経路を見て反応する悪意 | RangeMeanReversion, CrowdSqueeze, FalseBreakout, StopHunt |
| ANCHOR | 実データの骨格に合わせる | AnchorBridge |
| SPREAD | スプレッド | ConstantSpread, DynamicSpread |

- 部品は `params` で個別に上書きでき、`None` で外せる(`{"stop_hunt": {"prob": 0.1}, "crowd_squeeze": None}`)。
  未知のキーはエラーになる(打ち間違いで黙って既定値が使われるのを防ぐ)
- 部品間の受け渡しは `ctx.meta`(`regime`、`crowd_mix`、`news_idx`、`hunt_mask`、`anchor` など)
- 悪意のイベントは `ctx.event()` で記録され、SQLite の `market_events` と画面のマーカーになる
- `MarketData.model_info` に名前・版・全パラメータ・そのハッシュが入り、結果と一緒に保存される

### 実データ系

| モデル | 骨格 | 足の中 | シード |
|---|---|---|---|
| `real` | 実データ | 実データ(日足なら日足のみ) | 変わらない(1 本だけ回す) |
| `real_bridge` | 実データの足 | 季節性・ボラの塊・指標・裾の厚さを合成し、骨格の終値を必ず通る(OHLC があれば高値・安値にも届く) | 足の中が変わる |
| `real_bootstrap` | 実データのリターンをブロック単位で並べ替え | 同上 | 骨格も変わる |
| `real_bridge_malicious` | 実データの足 | 上 + 群衆の逆突き・ダマシ・ストップ狩り | 足の中が変わる |

足の中の振れ幅は、骨格の足ごとに「直前の局所ボラを中心にした事前分布」と「その足のリターン」から事後分布で引いた g に合わせ、
終値とのずれは事前の分散の時計で配る(`AnchorBridge`、v4 / R008)。足の中の荒れ方から日足のリターンの大きさが漏れないようにするため。

### 複数ペア(basket_*)

| モデル | 中身 |
|---|---|
| `basket_real` | 対円 7 ペアの実データ(FRED 日足、終値だけ)をそのまま再生 |
| `basket_bootstrap` | 日次リターンを全ペア同じブロック(同じ日付の並び)で並べ替える。ペアどうしの相関は残る |
| `basket_placebo` | 日次リターンに全ペア共通の日ごとの符号を掛ける(偽薬) |
| `basket_random_walk` | 実データの共分散を持つ正規乱数(ドリフト 0) |

生成した MarketData は先頭のペア(既定 usdjpy)のもので、全ペアは `meta["legs"]` に入る。1 ペアの戦略は先頭のペアで回り、
複数ペアの戦略(`Strategy.multi`)は `PortfolioBacktester` で全ペアを回す。足の中の経路は無い(日足の 2 点)。
検証(`fxsim verify`)は戦略が複数ペアなら、この 4 つを標準のパターンにし、帰無は `basket_direction_null`。

## バックテスト

- 足 i の終値で判断し、注文は足 i+1 の始値で約定(買いは ask、売りは bid)。先読みは構造上起きない
- 損切り・利確・ロスカットは、足の中の基準解像度の経路(bid/ask)で最初に触れたところで判定する(`intrabar="path"`)。
  点と点の間はブラウン橋とみなし、間で触れる確率も入れる(`touch="bridge"`、E007)。
  越えた 1 歩が飛び(`gap_k`)なら飛んだ先の価格で約定し、窓・フラッシュクラッシュでは損切り価格より大きく不利になる。
  ふつうの 1 歩なら損切り価格で約定する(1 分ごとの点で約定させると、刻みの分だけ毎回滑る)。損切りと利確が同じ足に入ったら、実際に先に届いた方。
  経路を持たない足(日足の再生など)は足の高安で判定し、両方届く足は損切りが先(悲観的)
- スプレッドが広がると bid の安値が下がるので、mid が届いていなくても損切りが刺さる(ロールオーバー・指標・薄商い)
- 素の損益(gross)= mid どうしの pip。費用の内訳(`metrics["costs"]`)= スプレッド + スリッページ + 手数料 − スワップ
- 破綻: 有効証拠金が一度でも初期の `ruin_frac`(既定 1 割)以下になったら `ruined`。損益は変えない。
  `stop_at_ruin: true` なら、その足で手仕舞いして取引をやめる(E007)
- スワップは政策金利の差から毎日(`fxsim/data/rates.py`)。ロスカットは証拠金維持率を割った点の価格で
- 一部決済(`ctx.reduce(units)`): 建玉の一部だけを次の足の始値で決済する(リピート系)。取得単価は平均のまま
- 複数ペア(`backtest/portfolio.py`): ペアごとの約定・費用・スワップ・損切りは 1 ペアのエンジンと同じ。残高・有効証拠金・
  建てられる量(有効証拠金 x レバレッジ − ほかのペアの建玉)は口座で共有する。ロスカットは口座全体を足の終値で見る。
  スワップはペアごとの金利差(`data/real/short_rates_monthly_fred.csv`)
- ブローカーの悪意(`malicious` プリセット): 不利な方向だけのスリッページ、利確の有利な窓を渡さない、
  逆指値の近くまで来たら確率で刺す、スワップとスプレッドの上乗せ、ラストルック、ゼロカットなし(docs/MARKET_MALICE.md)

## 保存(SQLite)

`fxsim/storage/db.py` の先頭にスキーマの説明がある。要点:

- `runs` 1 実験 / `paths` 1 経路(市場の案 x シード)/ `jobs` 1 バックテスト(経路 x ブローカー x 戦略)
- 計画の段階で全行を作るので、画面で全体の進み具合が見える
- バックテスト中は `flush_every` 本ごとに、`jobs` の進捗・残高、新しい `trades`、`equity` の点を書く
- 詳細保存のシード(`detail_seeds`)だけ、ローソク足・イベント・取引・細かい資産曲線を持つ。
  ほかのパターンは成績と粗い資産曲線だけ(DB を小さく保つため)
- 集計ビュー `v_market_summary` / `v_strategy_summary`。中央値などは `experiments/stats.py`(Python)と
  `web/lib/stats.ts`(画面)で同じ定義・同じキーで計算する。画面の API は「どのジョブがどのグループか」だけ返し、
  統計は選んだ指標ごとに画面側で取る
- 同じモデル・パラメータ・シード・期間の相場パターンは、中身(ローソク足・イベント)を `data_key` で 1 つだけ持ち、
  実験をまたいで共有する(`pattern_data`)
- WAL モード。Python が書いている間も画面は読み取り専用で読める。実験の終わりに WAL を本体へ書き戻す
- DB はコミットしない(手元だけ)。実験が終わったら成績の概要を `docs/results/<run_key>/` に書き出し、そちらをコミットする
  (`fxsim/reports/summary.py`)。概要は相場パターンの性質(paths.csv)とバックテストごとの成績(jobs.csv)と設定(run.json)で、
  資産曲線・取引・ローソク足は持たない。DB の番号は DB ごとに違うので、全環境で一意な `runs.run_key` で指す

## 画面(web/)

| パス | 内容 |
|---|---|
| `/` | 実験の一覧と進捗、全実験の戦略ランキング |
| `/runs/[id]` | 市場の性質、戦略の成績(パターンごとの統計と分布)、ジョブ一覧 |
| `/jobs/[id]` | ライブ表示: ローソク足 + 売買 + 悪意イベント、資産曲線、経過・取引回数・残高 |
| `/cases` | テストケースの一覧と最新の結果・判定 |
| `/cases/[id]` | 検証: 判定と根拠、パターンの組ごとの損益、資産曲線の束(中央値・10/90%・実データ・帰無)、年ごとの損益 |

ジョブの画面は 1 秒ごとに差分(最後の足以降・新しい取引・新しい資産曲線の点)だけを取りに行く。
