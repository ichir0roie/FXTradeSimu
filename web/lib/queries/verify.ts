import { all, one, parseJson, type Row } from "../db";
import type { CaseDef, CaseDetail, FanGroup, FanLine, FanPoint, Verdict, YearRow } from "../types";
import { initialBalance } from "./common";

const NULL_LABEL = "NULL";
// fxsim/verification.py の GROUPS と同じ
const GROUPS: Record<string, string> = {
  real: "real", real_bootstrap: "bootstrap", real_bridge: "bridge",
  real_bridge_placebo: "placebo",
  stylized: "synthetic", malicious: "synthetic", random_walk: "synthetic",
  real_bridge_malicious: "stress",
};
const GROUP_LABELS: Record<string, string> = {
  bootstrap: "並べ替え(独立)", bridge: "実データ + 日中ランダム", placebo: "偽薬", synthetic: "完全ランダム", stress: "ストレス",
};

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** 全ての線に共通する時刻ごとに、中央値と 10/90 パーセンタイル。 */
function band(lines: { points: { t: number; v: number }[] }[]): FanPoint[] {
  if (!lines.length) return [];
  const maps = lines.map((l) => new Map(l.points.map((p) => [p.t, p.v])));
  return lines[0].points
    .map((p) => p.t)
    .filter((t) => maps.every((m) => m.has(t)))
    .map((t) => {
      const v = maps.map((m) => m.get(t)!).sort((a, b) => a - b);
      return { t, median: quantile(v, 0.5), p10: quantile(v, 0.1), p90: quantile(v, 0.9) };
    });
}

export function caseDetail(testId: string): CaseDetail | null {
  const test = one(`SELECT id, kind, name, hash, definition_json FROM case_defs WHERE id = ?`, testId);
  if (!test) return null;
  const toDef = ({ definition_json, ...d }: Row) => ({ ...d, def: parseJson(definition_json, {}) }) as CaseDef;
  const def = toDef(test);
  const behavior = one(`SELECT id, kind, name, hash, definition_json FROM case_defs WHERE id = ?`, def.def.behavior ?? "");
  const cr = one(`SELECT run_id, status, summary_json FROM case_runs WHERE test_id = ? ORDER BY id DESC LIMIT 1`, testId);
  const empty: CaseDetail = {
    test: def, behavior: behavior ? toDef(behavior) : null, runId: null, runStatus: null,
    progress: { finished: 0, total: 0 }, verdict: null, fan: [], years: [], yearRows: [],
  };
  if (!cr?.run_id) return empty;
  const runId = Number(cr.run_id);
  const run = one(`SELECT status, config_json FROM runs WHERE id = ?`, runId);
  const init = initialBalance(run);
  const summary = parseJson<{ verdict?: Verdict }>(cr.summary_json, {});
  const jobs = all(
    `SELECT j.id, j.strategy_label, j.status, j.metrics_json, p.market, p.market_label, p.seed
     FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ? ORDER BY p.market_label, p.seed`,
    runId,
  );
  const finished = jobs.filter((j) => ["done", "skipped", "failed"].includes(j.status)).length;

  // 資産曲線(損益 = 有効証拠金 − 初期資金)
  const eq = all(
    `SELECT e.job_id, e.t, e.equity FROM equity e JOIN jobs j ON j.id = e.job_id WHERE j.run_id = ? ORDER BY e.job_id, e.t`,
    runId,
  );
  const series = new Map<number, { t: number; v: number }[]>();
  for (const r of eq) {
    if (!series.has(r.job_id)) series.set(r.job_id, []);
    series.get(r.job_id)!.push({ t: r.t, v: r.equity - init });
  }
  const model = summary.verdict?.model ?? jobs.find((j) => j.strategy_label !== NULL_LABEL)?.strategy_label;
  const fan: FanGroup[] = Object.keys(GROUP_LABELS)
    .map((key) => {
      const inGroup = jobs.filter(
        (j) => ((GROUPS[j.market] ?? "synthetic") === key || j.market === "real") && series.has(j.id),
      );
      const lines: FanLine[] = inGroup
        .filter((j) => j.strategy_label === model)
        .map((j) => ({ jobId: j.id, market: j.market_label, seed: j.seed, real: j.market === "real", points: series.get(j.id)! }));
      const nulls = inGroup
        .filter((j) => j.strategy_label === NULL_LABEL && j.market !== "real")
        .map((j) => ({ points: series.get(j.id)! }));
      const synthetic = lines.filter((l) => !l.real);
      return {
        key,
        label: GROUP_LABELS[key],
        lines,
        band: band(synthetic.length ? synthetic : lines),
        nullMedian: band(nulls).map((p) => ({ t: p.t, v: p.median })),
      };
    })
    .filter((g) => g.lines.length);

  // 年ごとの損益(モデル、実データ系)
  const yearRows: YearRow[] = jobs
    .filter((j) => j.strategy_label === model && ["real", "bootstrap"].includes(GROUPS[j.market] ?? "") && j.status === "done")
    .map((j) => ({
      label: `${j.market_label} #${j.seed}`,
      market: j.market_label,
      seed: j.seed,
      real: j.market === "real",
      values: parseJson<{ yearly_pnl?: Record<string, number> }>(j.metrics_json, {}).yearly_pnl ?? {},
    }));
  const years = [...new Set(yearRows.flatMap((r) => Object.keys(r.values)))].sort();
  return {
    ...empty,
    runId,
    runStatus: run?.status ?? null,
    progress: { finished, total: jobs.length },
    verdict: summary.verdict ?? null,
    fan,
    years,
    yearRows,
  };
}
