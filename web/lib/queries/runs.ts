import { all, one, parseJson } from "../db";
import type { CostRow, JobGroup, JobRow, PathRow, RunDetail, RunListItem, StrategySummaryRow } from "../types";
import { initialBalance, livePnl } from "./common";

export function listRuns(): RunListItem[] {
  return all(`
    SELECT r.id, r.name, r.kind, r.proposal, r.status, r.created_at, r.git_sha, r.years, r.elapsed_sec,
           (SELECT COUNT(*) FROM jobs j WHERE j.run_id = r.id) AS n_jobs,
           (SELECT COUNT(*) FROM jobs j WHERE j.run_id = r.id AND j.status IN ('done','skipped','failed')) AS n_finished,
           (SELECT COUNT(*) FROM paths p WHERE p.run_id = r.id) AS n_paths,
           (SELECT SUM(COALESCE(j.bars_done,0)) * 1.0 / NULLIF(SUM(COALESCE(j.bars_total,0)),0)
              FROM jobs j WHERE j.run_id = r.id AND j.status != 'skipped') AS bar_progress
    FROM runs r ORDER BY r.id DESC`) as RunListItem[];
}

export function strategyRanking(limit = 20): StrategySummaryRow[] {
  return all(
    `SELECT * FROM v_strategy_summary WHERE strategy != 'random_entry' ORDER BY sharpe DESC LIMIT ?`,
    limit,
  ) as StrategySummaryRow[];
}

/** (ブローカー, 戦略, 市場) と、市場をまとめた ALL ごとに、完了したジョブを束ねる。統計は画面側で取る。 */
function groupJobs(jobs: JobRow[]): JobGroup[] {
  const groups = new Map<string, JobGroup>();
  for (const j of jobs) {
    if (j.status !== "done") continue;
    for (const market of [j.market_label, "ALL"]) {
      const k = `${j.broker}\u0000${j.strategy_label}\u0000${market}`;
      if (!groups.has(k)) groups.set(k, { broker: j.broker, strategy: j.strategy_label, market, jobIds: [] });
      groups.get(k)!.jobIds.push(j.id);
    }
  }
  return [...groups.values()].sort(
    (a, b) =>
      a.broker.localeCompare(b.broker) ||
      a.strategy.localeCompare(b.strategy) ||
      Number(a.market === "ALL") - Number(b.market === "ALL") ||
      a.market.localeCompare(b.market),
  );
}

/** ブローカー x 戦略ごとの費用の内訳(円、パターン平均)。fxsim/storage/queries.py の cost_summary と同じ。 */
function costSummary(id: number): CostRow[] {
  return all(
    `SELECT j.broker, j.strategy_label, COUNT(*) AS n,
            AVG(json_extract(j.metrics_json, '$.costs.gross_pnl')) AS gross_pnl,
            AVG(json_extract(j.metrics_json, '$.costs.spread')) AS spread,
            AVG(json_extract(j.metrics_json, '$.costs.slippage')) AS slippage,
            AVG(json_extract(j.metrics_json, '$.costs.commission')) AS commission,
            AVG(json_extract(j.metrics_json, '$.costs.swap')) AS swap,
            AVG(j.pnl) AS pnl,
            AVG(json_extract(j.metrics_json, '$.hunted')) AS hunted,
            AVG(json_extract(j.metrics_json, '$.losscut')) AS losscut,
            AVG(json_extract(j.metrics_json, '$.rejected')) AS rejected,
            AVG(json_extract(j.metrics_json, '$.zero_cut')) AS zero_cut
     FROM jobs j WHERE j.run_id = ? AND j.status = 'done'
       AND json_extract(j.metrics_json, '$.costs') IS NOT NULL
     GROUP BY j.broker, j.strategy_label ORDER BY j.broker, j.strategy_label`,
    id,
  ) as CostRow[];
}

export function runDetail(id: number): RunDetail | null {
  const run = one(`SELECT * FROM runs WHERE id = ?`, id);
  if (!run) return null;
  const init = initialBalance(run);
  const paths = all(
    `SELECT id, market, market_label, seed, detail, status, realism_score, n_fail, facts_json
     FROM paths WHERE run_id = ? ORDER BY market_label, seed`,
    id,
  ).map(({ facts_json, ...p }) => ({ ...p, facts: parseJson(facts_json, {}) })) as PathRow[];
  const jobs = all(
    `SELECT j.id, j.path_id, j.broker, j.strategy, j.strategy_label, j.timeframe, j.status, j.note,
            j.bars_total, j.bars_done, j.equity, j.n_trades,
            j.pnl, j.sharpe, j.gross_sharpe, j.max_dd, j.net_pips_per_year, j.trades_per_year, j.win_rate,
            p.market_label, p.market, p.seed, p.detail
     FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ?
     ORDER BY j.broker, j.strategy_label, p.market_label, p.seed`,
    id,
  ).map((j) => ({ ...j, pnl_live: livePnl(j, init) })) as JobRow[];
  const finished = jobs.filter((j) => j.status === "done" || j.status === "skipped" || j.status === "failed").length;
  return {
    run: {
      id: run.id,
      name: run.name,
      kind: run.kind,
      status: run.status,
      years: run.years,
      seeds_json: run.seeds_json,
      git_sha: run.git_sha,
      proposal: run.proposal,
    },
    initialBalance: init,
    paths,
    jobs,
    groups: groupJobs(jobs),
    costs: costSummary(id),
    progress: { finished, total: jobs.length },
  };
}
