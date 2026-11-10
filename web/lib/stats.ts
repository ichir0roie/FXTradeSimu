// パターン(シード・実データ)ごとの結果の統計。Python の fxsim/experiments/stats.py と同じ定義・同じキー。
import type { Summary } from "./types";

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function summarize(values: (number | null | undefined)[]): Summary | null {
  const v = values.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const total = v.reduce((a, b) => a + b, 0);
  const mean = total / v.length;
  const std = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : 0;
  return {
    n: v.length,
    mean,
    median: quantile(s, 0.5),
    total,
    std,
    min: s[0],
    max: s[s.length - 1],
    p10: quantile(s, 0.1),
    p90: quantile(s, 0.9),
    positive_ratio: v.filter((x) => x > 0).length / v.length,
    // パターン単位の t 値(|t| ≥ 2 が有意の目安)
    t: v.length > 1 && std > 0 ? mean / (std / Math.sqrt(v.length)) : 0,
  };
}
