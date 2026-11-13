# FXTradeSimu

FX の売買の方法を、「ありえた別の歴史」も含めた多数の相場で長期間シミュレーションし、本当に勝てるのかを確かめるための道具と、
その調査の記録。後半からは株価指数の積立・レバレッジ・税(NISA)などの投資の試算も扱う。

**結果のまとめは [`docs/FINDINGS.md`](docs/FINDINGS.md)。** 要点だけ言うと、個人の口座で値動きを読んで勝てる単純な規則は
見つからず、残ったのはスワップ(金利差)を受け取る向きに数週間だけ乗る型だけだった。

> 投資の助言ではない。過去のデータとシミュレーションでの結果で、将来の成績を約束しない。

## できること

- **市場モデル**: ランダムウォークから、実際の為替の統計的性質を持つ相場、さらに「群衆が一番痛がる方へ動く」悪意
  (ストップ狩り・ダマシ・群衆の逆突き)を持つ相場まで。部品を組み合わせて案を増やせる
- **実データ**: USD/JPY と対円 7 ペアの日足・8 通貨の短期金利(FRED)をそのまま再生するほか、実データを通りつつ足の中だけ
  ランダムな相場や、実データのリターンを並べ替えた相場を何本でも作れる
- **異常な値動きと薄商い**: フラッシュクラッシュ・為替介入・ショック、年末年始などの薄商い(スプレッドが広がる)
- **取引の費用**: スプレッド、手数料(ECN)、金利差から毎日計算するスワップ、足の中の経路で約定する損切り(急変では大きく滑る)、
  ロスカット。費用は内訳(スプレッド・滑り・手数料・スワップ)で記録
- **業者の悪意**: 不利な方向だけの滑り、逆指値を狙うストップ狩り、スワップとスプレッドの上乗せ、ラストルック(有利な注文の拒否)
- **よくある手法**: デイトレード・テクニカル(RSI・MACD・一目・ボリンジャーなど)・中長期(タートル・キャリーなど)・
  学習するモデル・賭け方(マーチンゲール・ナンピン・ケリーなど)。一覧は `docs/proposals/strategies/`
- **長期の検証**(`fxsim verify`): 実データ・並べ替えた別の歴史・偽薬・完全ランダムの多数のパターンで、「向きだけでたらめな版」と
  対にして回し、長期的に利益が出るかを判定する
- **実データだけでの検証**(`fxsim pair-eval`): 主要 10 ペアを期間に切り、向きだけでたらめな版と対にして回す
- **MT5 での常駐**(`fxsim trade`、任意): 検証した戦略を、同じコードのまま MT5 のデモ口座で動かす(`docs/TRADING.md`)

結果は手元の SQLite(`data/fxsim.sqlite`)に入る。乱数のシードで結果が決まるので、同じ設定を回せば同じ結果が出る。

## 使い方

Python 3.10 以上(手元では 3.12 で確かめている)。

```bash
pip install -r requirements.txt
python -m pytest -q                                   # テスト
python -m fxsim list                                  # 市場モデル・戦略・業者
python -m fxsim run configs/quick.yaml                # 動作確認(数十秒)
python -m fxsim report --patterns                     # 最新の実験の結果を Markdown で
python -m fxsim eval-strategy ma_cross --params '{"timeframe": 1440}' --null --patterns
python -m fxsim verify B049                           # ひとつの戦略の 30 年の長期検証(数十分)
python -m fxsim eval-market malicious                 # 市場の案の評価
python -m fxsim.research.leveraged_index              # 株価指数の 1 倍・NISA と 2 倍型の試算(E059)
```

実験ごとの研究のスクリプトは `fxsim/research/` に 1 本ずつあり、提案書の「測り方」にコマンドが書いてある。
1 時間足以下の実データ(業者の足)は再配布できないのでリポジトリに入れていない。MT5 の口座があれば
`python -m fxsim fetch-mt5 configs/market_data/oanda-demo.yaml` で手元の DB(`data/market.sqlite`)に溜められる。

## ドキュメント

- [`docs/FINDINGS.md`](docs/FINDINGS.md) — 調査結果のまとめ
- `docs/INSIGHTS.md` — 知見の一覧(実験ごとの細かい数字)
- `docs/CANDIDATES.md` — 戦略の候補と、その検証の状態
- `docs/proposals/` — 案ごとの仮説・設計・結果・判定・次の一手(市場 `M`・`R`、戦略 `S`、実験 `E`)
- `docs/ARCHITECTURE.md` — コードの構成
- `docs/MARKET_MALICE.md` — 市場の悪意の一覧と測り方
- `docs/CASES.md` — 動作ケースとテストケースの書き方
- `docs/TRADING.md` — MT5 の口座で戦略を常駐させる
- `data/real/README.md` — 実データの出典とライセンス
- `CLAUDE.md` — このリポジトリで Claude Code が案を出し、評価し、記録する手順

## データの出典

- FRED(米連邦準備制度理事会 H.10 の為替、OECD の短期金利)。詳しくは `data/real/README.md`
- `data/real/policy_rates_usd_jpy.csv` は公開情報からの手入力の概算で、誤差がある

## ライセンス

MIT([`LICENSE`](LICENSE))。`data/real/` のデータは各出典の条件に従う。

## 履歴

- 2021/02 ランダムウォークのチャートを PDF で出力する最初の版
- 2026/11 `fxsim` パッケージとして作り直し、売買の方法の検証を進めた。公開にあたり、手元専用の画面(Next.js)は外した
