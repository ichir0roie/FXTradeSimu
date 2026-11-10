# R007 実データ系に重ねるフラッシュクラッシュと薄商い(real_bridge 系 v3)

- 種類: 市場モデル / 状態: 採用
- 実装: `fxsim/markets/components/extreme.py` の `FlashCrashOverlay`(登録名 `flash_overlay`、ANCHOR 段)、`HolidayLiquidity`
- 対象: real_bridge / real_bridge_malicious / real_bootstrap / real_bridge_placebo(すべて v3)
- 親: R005(real_* v2)、M005

## 仮説
実データ系は日足の終値を実データに合わせるので、M005 の `extreme` をそのまま入れると AnchorBridge が急変を均してしまい、
介入やショックは実データの日足に既に入っている分と二重になる。
**その日のうちに完全に戻るフラッシュクラッシュ**だけなら、日足の終値を変えずに足の中に足せる。
戦略にとっては、損切りが急落の底で約定して戻りを取れない、という痛みだけが加わる。

## 設計
- `flash_overlay`: `ExtremeEvents` の急落だけを残し、戻る割合を 100%、戻りの時定数を 15 分〜3 時間にした版。
  AnchorBridge の後(同じ ANCHOR 段で、並べた順)に動く
- 終わりの位置を「その取引日の最後の足」と「骨格の次の終値の位置」の手前に制限する(`_limits`)。
  終わりで値はちょうど 0 に戻るので、日足の終値(骨格の点)は 1e-12 の精度で変わらない。戻る余地の無い時刻に当たったものは起こさない
- 日足の高値・安値は実データを越えることがある(FRED の終値だけのデータでは高値・安値はもともと合成)
- `holidays` は足の中の部品(`_intraday`)に入れる。スプレッドが広がり、ボラが少し下がる(終値への合わせ込みでボラの大きさは実データに戻る)

v2 の再現: `{"holidays": null, "flash_overlay": null}`。

## 測り方
```bash
python -m fxsim eval-market real_bridge --years 10 --seeds 1-8 --proposal R007
python -m fxsim eval-market real_bridge --params '{"holidays": null, "flash_overlay": null}' --label real_bridge_v2 --years 10 --seeds 1-8 --proposal R007
```
- 日足の終値が v2 と同じ(`tests/test_extreme.py::test_flash_overlay_keeps_daily_closes_of_the_real_skeleton`)
- 素のシャープが v2 から大きく変わらない

## 結果(10 年 x 8 パターン、fair ブローカー、素のシャープ。括弧はパターン単位の t)

| 戦略 | real_bridge v2 | **real_bridge v3** |
|---|---|---|
| bollinger_reversion | −0.57 (−10.97) | −0.48 (−9.70) |
| donchian_breakout | +0.03 (−0.54) | +0.01 (−0.90) |
| ma_cross | +0.03 (−0.22) | −0.01 (−1.40) |
| simple_follower | +0.03 (−1.64) | +0.06 (−1.45) |
| random_entry | −0.32 (−3.95) | −0.32 (−4.15) |

現実らしさ 100 → 100、分散比 H1→D1 1.002 → 0.982、急落は 0.23 回/年(戻る余地が無くて起こさなかった分だけ M005 より少ない)。
bollinger_reversion の大きな負けは v2 から同じで、8 パターンが同じ実データの日足を共有するため(この 10 年の強いトレンドに逆らう)。
パターンは日足では独立でないので、この t は大きく見えすぎる(E002 の擬似反復)。

## 判定
採用。日足の終値を保ったまま、足の中に急落と薄商いを加えられた。素のシャープはほとんど変わらない。
完全に戻る急落は本物の平均回帰なので、分散比はわずかに下がる(1 回で大きく動く。R005 の検査は急落を外して測る)。

## 次の一手
- 実データの日足に入っている介入の日(2022-09-22、2022-10-21、2024-04-29、2024-05-01、2024-07-11)の足の中を、
  介入の形(数十分で一方向に段で動く)で作る。今は普通の日と同じ合成
