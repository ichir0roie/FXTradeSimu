# E007 端末による再現性、損切りの約定の偏り、破綻の数え方(と v4 での長期の回し直し)

- 種類: 実験 / 状態: 実行済み
- 実装: `fxsim/markets/components/shocks.py`(安定ソート)、`fxsim/backtest/engine.py`(`_first_touch`、破綻)、
  `fxsim/backtest/broker.py`(`touch`)、`fxsim/backtest/engine.py::EngineConfig`(`ruin_frac`、`stop_at_ruin`)
- 設定: `configs/long_run_v4.yaml`(E005 と同じ組み合わせを市場 v4・エンジンの修正後で)
- 関連: R008(real_* v4)、M006(malicious v4)

## 見つけたこと 1: 同じシードでも端末によって相場が変わっていた
DB の run #2(E004、クラウドで実行)を手元(AMD Ryzen、AVX2 のみ)で同じ設定のまま回すと、576 本中 550 本の損益が変わった。
コードの差ではない(E004 のときのコードと今のコードで、市場の部品は同じ)。

- random_walk と real_bridge は 1e-12 まで一致し、stylized と malicious だけが変わる
- 手元で numpy の SIMD を切ると(`NPY_DISABLE_CPU_FEATURES`)、malicious seed 1 の尖度 H1 が 83.15 → 117.91 に変わる(クラウドは 143.25)
- 原因は `NewsShocks` の `np.argsort(idx_list)`。既定のクイックソートは安定ではなく、雇用統計とランダムな指標が同じ分に
  重なったとき、どちらの大きさ(35 pip か 12 pip)が付くかが、numpy の SIMD のソートの実装(AVX2 / AVX-512)で変わる。
  1 つずれると後の悪意(相場を見て動く部品)が全部変わる
- `kind="stable"` に直すと、SIMD の有無で乱数の状態・イベントの数が一致し、値の差は 1e-10 程度の丸めだけになる
  (malicious・stylized・real_bridge_malicious、10 年 x 3 シード)
- 回帰テスト: `tests/test_markets.py::test_markets_do_not_depend_on_the_cpu_simd_features`(SIMD を切った子プロセスと比べる。
  古いソートに戻すと落ちることを確かめた)

影響: これより前の DB の結果(E004〜E006、T001〜T006、V-B004 / V-B007)は、クラウドの CPU で作った相場で、手元では再現しない。
並べ替えを直すと値動きが変わるので、NewsShocks を使うモデルの版を上げた(stylized v3、malicious v4、real_* v4)。
古い版の結果は DB に残している(版が違うので、相場パターンの保存のキーもぶつからない)。

## 見つけたこと 2: 損切りを使う戦略がランダムウォークでも得をしていた
E004 で「点と点の間も連続して動く」として、1 分の点で損切りを越えたら損切りの価格で約定させるようにした。
ところが、点の間で損切りに触れて点に現れる前に戻った場合は損切りにならない。結果として、越えたら損切りの価格、
戻ったら持ち続け、を後から選べることになり、損切りを使う側にただのオプションを渡していた。

ランダムウォーク(5 年 x 96 パターン、fair、素の pip/年)

| 戦略 | 点で判定・損切りの価格(E004〜E006) | 点で判定・越えた点の価格 | **ブラウン橋で判定・損切りの価格(今回)** |
|---|---|---|---|
| rsi_reversion | +79 (2.7) | +19 (0.6) | +16 (0.6) |
| bollinger_reversion | +109 (3.2) | −0 (−0.0) | −12 (−0.4) |
| stochastic_reversion | +96 (3.3) | −8 (−0.3) | −2 (−0.1) |
| donchian_breakout | +67 (1.7) | +3 (0.1) | +4 (0.1) |
| random_entry | +47 (1.8) | +47 (1.8) | +33 (1.3) |
| ma_cross | −38 (−0.8) | −52 (−1.0) | −53 (−1.1) |

直し方: 点と点の間をブラウン橋とみなし、両端が損切り(利確)の手前にあっても、間で触れる確率
exp(−2ab/σ²)(a, b は両端から水準までの距離、σ は 1 歩の標準偏差を中央値から見積もったもの)で触れたことにする。
触れたら水準の価格で約定する(連続の解釈と矛盾しない)。飛び(|1 歩| > gap)は拡散ではないので対象外で、越えた先で約定する。
乱数はブローカー用とは別の系列(`touch_rng`)。`BrokerConfig.touch = "points"` で E004〜E006 の判定に戻せる。
テスト: `tests/test_costs.py::test_bridge_touch_probability_matches_the_brownian_bridge`。

影響: 損切りを使う戦略の素の損益は、これまで年 60〜110 pip ほど甘かった。E001〜E006 の「逆張りが勝つ穴」の一部はこれ。

## 見つけたこと 3: 破綻が数えられていなかった(E005 の次の一手)
建てる量は「有効証拠金 x レバレッジ」で頭打ちになるので、破綻しかけた口座が少額で取引を続け、有効証拠金が 0 にならない。
`ruined` は 0 以下になったときしか立たず、E005・E006 で −99.6 万円の口座も「破綻 0」だった。

- `EngineConfig.ruin_frac`(既定 0.1): 有効証拠金が初期の 1 割以下に一度でもなれば破綻として数える。損益は変わらない
  (同じ設定を回して 576 本で 1 円も変わらないことを確かめた)
- `EngineConfig.stop_at_ruin`(既定 False): True なら破綻した足で手仕舞いして、それ以降は取引しない(決済の理由は `ruin`)
- 成績に `ruin_years`(破綻までの年数)、`min_equity_frac`、`blocked`(証拠金不足で建てられなかった注文の数)を足した
- report の戦略の表に「破綻」(破綻したパターンの割合)、検証の表にも同じ列を足した

## 測り方
```bash
python -m pytest -q
python -m fxsim run configs/long_run_v4.yaml
python -m fxsim report <run>
```
E005 の表と比べる。

## 結果
(回し直しの結果を下に追記)

## 次の一手
- runs に CPU の SIMD の情報を記録する(再現しないときにすぐ分かるように)
- 並べ替え以外にも、同じ値の扱いが実装に依存する箇所が無いか(`np.unique`、`argmax` の同点など)を定期的に見る。
  回帰テストは malicious と real_bridge_malicious の 1 年だけを見ている
