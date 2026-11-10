# 実験の概要

1 実行 = 1 フォルダ(`<run_key>/`)。コミットする結果はこれだけで、DB(`data/fxsim.sqlite`)は手元だけに置く。
既定の DB への実験(demo 以外)は、終わったら自動でここに書き出される。書き出し直すなら `python -m fxsim summary <run_key>`。

| ファイル | 中身 |
|---|---|
| `run.json` | 実験の名前・種類・提案 ID・コードの版(`git_sha`)・期間・シード・設定の全体(`config`)。テストケースと検証なら、定義のハッシュと要約・判定(`case_runs`) |
| `paths.csv` | 相場パターン(市場の案 x シード)ごとに 1 行。期間、現実らしさ(`realism_score`、`n_fail`)、統計的性質(facts) |
| `jobs.csv` | バックテスト(パターン x ブローカー x 戦略)ごとに 1 行。損益(`pnl`)・シャープ・素のシャープ・最大 DD・取引回数などの成績、費用の内訳(`cost_*`)、決済の理由(`exit_reasons`)と年ごとの損益(`yearly_pnl`)は JSON |

- 資産曲線・取引の明細・ローソク足・悪意のイベントは持たない。結果はシードで決まるので、要るときは
  `run.json` の `config` を同じコードの版で回し直す
- パターン単位の統計(平均・中央値・t 値・プラスの割合)は jobs.csv から取り直せる。
  `jobs.csv` の `path_id` が `paths.csv` の `path_id` に対応する

```python
from fxsim.reports import read_summary
s = read_summary("E006")            # 前方一致も可(1 つに決まるとき)
pnl = [j["pnl"] for j in s["jobs"] if j["strategy_label"] == "donchian_breakout" and j["broker"] == "malicious"]
```

## 番号の対応

提案書の「run #N」は、書いた時点の DB の番号。DB は環境ごとに違うので、run_key で引き直す。
2026-11-10 の結果パック導入より前の実行は、run_key の末尾 `rNN` がその頃の番号。

| 旧番号 | run_key | 中身 |
|---|---|---|
| #1 | E005-20261110T045632-r01 | E005 長期シミュレーション(市場 v3) |
| #2 | E004-20261110T045938-r02 | E004 費用 |
| #3〜#8 | T001〜T006-20261110T05xxxx-r03〜r08 | テストケース T001〜T006 |
| #9 | V-B007-20261110T050142-r09 | B007 の長期検証 |
| #10 | V-B004-20261110T050414-r10 | B004 の長期検証 |
| #12 | E006-20261110T050654-r12 | E006 よくある投資手法 |

これらはクラウドの CPU(AVX-512)で作った相場で、手元(AVX2)では同じ相場にならない(E007。直した後の版は
CPU によらない)。
