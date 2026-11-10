# 実際の相場データ

| ファイル | 内容 | 出典 / ライセンス |
|---|---|---|
| `usdjpy_d1_fred.csv` | USD/JPY 日次(NY 正午の買いレート、終値のみ)1971-01-04 〜 2026-10-30 | FRED(米連邦準備制度理事会 H.10)を [datasets/exchange-rates](https://github.com/datasets/exchange-rates) 経由で取得。ODC-PDDL(パブリックドメイン) |

## 自分のデータを足す

`fxsim.markets.real.load_ohlc` が読める CSV なら、ここに置いて市場モデルの `source.path` に指定すれば使える。

- 列: `time` と `close` は必須。`open` `high` `low` があれば OHLC として扱う(無ければ終値だけ)
- `time` は UTC。日付だけ(`2024-01-05`)なら、その日の NY クローズ(22:00 UTC)の値とみなす
- HistData の Generic ASCII(`20240105 170000;146.10;146.12;146.08;146.11;0`、EST 固定)は `format: histdata` で読める

1 分足を置けば `real`(そのまま再生)を 1 時間足の戦略で使える。日足しか無い場合、
1 時間足の戦略は `real_bridge` / `real_bootstrap`(日足の骨格に日中の動きを合成)で評価する。
