# 設計

```
configs/*.yaml ──> fxsim.experiments.runner ──> data/fxsim.sqlite <── web/(Next.js、ローカル専用)
                         │                             ▲
                         ├─ markets: 経路を作る ───────┤ paths / candles / market_events
                         └─ backtest: 売買を回す ──────┘ jobs / equity / trades(途中経過も逐次)
```

## パッケージ構成

```
fxsim/
  calendar.py          取引カレンダー(UTC、週は日曜 22:00〜金曜 22:00、ロールオーバー 22:00)
  data.py              MarketData(基準解像度の mid とスプレッド)と Bars(任意の時間足、bid/ask の OHLC)
  numerics.py          AR(1) フィルタ、t 分布など
  registry.py          名前 -> ファクトリ(市場モデル・戦略で共用)
  catalog.py           組み込みの案をすべて登録する
  markets/
    base.py            Stage / Component / GenContext / MarketModel
    components/        部品(volatility, regime, shocks, malice, real, spread)
    models/            案(random_walk, stylized, malicious, real_based)
    real.py            実データの読み込み
    evaluation.py      現実らしさ(stylized facts)と悪意の指標
  backtest/
    broker.py          ブローカー(スリッページ、スワップ、ロスカット、ストップ狩り)
    engine.py          1 ポジションのバックテスト
    metrics.py         成績
    lookahead.py       先読み検査
  strategies/
    base.py            Strategy(prepare / on_bar)
    indicators.py      因果的な指標
    builtin/           案(1 ファイル = 1 戦略)
  storage/db.py        SQLite のスキーマ・ビュー
  experiments/
    config.py          YAML の実験設定
    runner.py          経路ごとに 1 プロセス、途中経過を逐次書き込み
    stats.py           パターンごとの統計(平均・中央値・合計・標準偏差・最小/最大・プラスの割合)
    report.py          Markdown の表
  cli.py               python -m fxsim ...
web/                   Next.js 16(App Router)+ better-sqlite3 + lightweight-charts
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

足の中の振れ幅は、骨格に固定しなかった場合の日次の分散が実データと同じになるよう自動で伸縮する(`AnchorBridge.calibrate_vol`)。

## バックテスト

- 足 i の終値で判断し、注文は足 i+1 の始値で約定(買いは ask、売りは bid)。先読みは構造上起きない
- 損切り・利確は足の bid/ask の高値・安値で判定。両方届く足は損切りが先(悲観的)。窓は始値で約定
- スプレッドが広がると bid の安値が下がるので、mid が届いていなくても損切りが刺さる(ロールオーバー・指標)
- 素の損益(gross)= mid どうしの pip。コスト = gross − 実際の pip(平均スプレッド + スリッページ)
- ブローカーの悪意(`malicious` プリセット): 不利な方向だけのスリッページ、利確の有利な窓を渡さない、
  逆指値の近くまで来たら確率で刺す、スワップの上乗せ

## 保存(SQLite)

`fxsim/storage/db.py` の先頭にスキーマの説明がある。要点:

- `runs` 1 実験 / `paths` 1 経路(市場の案 x シード)/ `jobs` 1 バックテスト(経路 x ブローカー x 戦略)
- 計画の段階で全行を作るので、画面で全体の進み具合が見える
- バックテスト中は `flush_every` 本ごとに、`jobs` の進捗・残高、新しい `trades`、`equity` の点を書く
- 詳細保存のシード(`detail_seeds`)だけ、ローソク足・イベント・取引・細かい資産曲線を持つ。
  ほかのパターンは成績と粗い資産曲線だけ(DB を小さく保つため)
- 集計ビュー `v_market_summary` / `v_strategy_summary`。中央値などは `experiments/stats.py`(Python)と
  `web/lib/stats.ts`(画面)で同じ定義で計算する
- WAL モード。Python が書いている間も画面は読み取り専用で読める。実験の終わりに WAL を本体へ書き戻す

## 画面(web/)

| パス | 内容 |
|---|---|
| `/` | 実験の一覧と進捗、全実験の戦略ランキング |
| `/runs/[id]` | 市場の性質、戦略の成績(パターンごとの統計と分布)、ジョブ一覧 |
| `/jobs/[id]` | ライブ表示: ローソク足 + 売買 + 悪意イベント、資産曲線、経過・取引回数・残高 |

ジョブの画面は 1 秒ごとに差分(最後の足以降・新しい取引・新しい資産曲線の点)だけを取りに行く。
