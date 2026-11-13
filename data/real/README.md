# 実際の相場データ

| ファイル | 内容 | 出典 / ライセンス |
|---|---|---|
| `policy_rates_usd_jpy.csv` | 米国(FF 金利の誘導目標のおおよその中心)と日本(無担保コール翌日物の誘導目標・政策金利のおおよその値)の政策金利、変更日ごと(年率 %) | **手入力の概算**。Claude が公開情報の記憶から作ったもので、日付・値とも誤差がある。1990 年以前は数年ごとの代表値。2025-12 以降の変更は入っていない可能性がある。スワップの計算に使う。正確に使うなら FRED の `DFF` と日銀の時系列で差し替える(列: date, usd, jpy) |
| `fx_jpy_crosses_d1_fred.csv` | 対円 7 ペア(usdjpy, eurjpy, gbpjpy, audjpy, nzdjpy, cadjpy, chfjpy)の日次の終値、1999-01-04 〜。全ペアの値がそろう日だけ。対ドルの H.10(NY 正午の買いレート)から掛け算で作った(EURJPY = EURUSD x USDJPY など) | FRED(DEXJPUS, DEXUSEU, DEXUSUK, DEXUSAL, DEXUSNZ, DEXCAUS, DEXSZUS)。パブリックドメイン。`python -m fxsim fetch-fred` で取り直せる |
| `short_rates_monthly_fred.csv` | 8 通貨(usd, jpy, eur, gbp, aud, nzd, cad, chf)の短期金利の月次(年率 %)。3 か月物の銀行間金利(OECD MEI の IR3TIB01)、その系列が始まる前はコール金利(IRSTCI01)で埋め、終わった後は最後の値のまま。月 M の平均を翌月 1 日から有効にしている(判断に未来の値を使わないため) | FRED 経由の OECD Main Economic Indicators。`python -m fxsim fetch-fred`。複数ペアのスワップとキャリーに使う |
| `usdjpy_d1_fred.csv` | USD/JPY 日次(NY 正午の買いレート、終値のみ)1971-01-04 〜 2026-10-30 | FRED(米連邦準備制度理事会 H.10)を [datasets/exchange-rates](https://github.com/datasets/exchange-rates) 経由で取得。ODC-PDDL(パブリックドメイン) |

## 自分のデータを足す

`fxsim.markets.real.load_ohlc` が読める CSV なら、ここに置いて市場モデルの `source.path` に指定すれば使える。

- 列: `time` と `close` は必須。`open` `high` `low` があれば OHLC として扱う(無ければ終値だけ)
- `time` は UTC。日付だけ(`2024-01-05`)なら、その日の NY クローズ(22:00 UTC)の値とみなす
- HistData の Generic ASCII(`20240105 170000;146.10;146.12;146.08;146.11;0`、EST 固定)は `format: histdata` で読める

## MT5 から溜めた足(相場の DB)

MT5 の口座から取った足は、CSV ではなく相場の DB(`data/market.sqlite`、コミットしない)に溜める。
業者の値は再配布できるか分からないので、リポジトリには入れない。

```bash
python -m fxsim fetch-mt5 configs/market_data/oanda-demo.yaml   # 書き足す(同じ足は上書き。何度流してもよい)
python -m fxsim market-data                                     # 溜まっている足の一覧
```

- テーブルは時間足ごと(`rates_m1` / `rates_m5` / ... / `rates_d1`)。1 行 = 出どころ(MT5 のサーバー名)x 銘柄 x 足
- 時刻(`time`)は足の始まりの UTC。MT5 のサーバー時刻は業者の決まり(OANDA Japan は NY + 7 時間、夏 UTC+3 / 冬 UTC+2)で
  UTC に直す。元のサーバー時刻は `server_time` に控えてある
- 価格は bid、`spread` はポイント。読むときは足ごとのスプレッドの半分を足して mid にする
- 端末が遡れるのは 10 万本の少し手前まで(1 分足で 3 か月ほど)。それより間を空けると、その間の足は二度と取れない
  (`fetches.connected` が 0 になり、警告が出る)
- 読むときは、実データのパス(市場モデルの `source.path`、`verify --dataset`)に `db:USDJPY/M1` と書く。
  出どころが複数あるなら `db:OANDA-Japan MT5 Demo/USDJPY/M1`。日足(`db:USDJPY/D1`)の終わりは 22:00 UTC に揃える

1 分足を置けば `real`(そのまま再生)を 1 時間足の戦略で使える。日足しか無い場合、
1 時間足の戦略は `real_bridge` / `real_bootstrap`(日足の骨格に日中の動きを合成)で評価する。
