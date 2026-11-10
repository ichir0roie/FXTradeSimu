import { all, one, parseJson, Row } from "./db";
import { summarize, Summary } from "./stats";

const DONE = ["done", "skipped", "failed"];

export function listRuns(): Row[] {
  return all(`
    SELECT r.id, r.name, r.kind, r.proposal, r.status, r.created_at, r.finished_at, r.git_sha, r.years,
           r.seeds_json, r.elapsed_sec,
           (SELECT COUNT(*) FROM jobs j WHERE j.run_id = r.id) AS n_jobs,
           (SELECT COUNT(*) FROM jobs j WHERE j.run_id = r.id AND j.status IN ('done','skipped','failed')) AS n_finished,
           (SELECT COUNT(*) FROM paths p WHERE p.run_id = r.id) AS n_paths,
           (SELECT SUM(COALESCE(j.bars_done,0)) * 1.0 / NULLIF(SUM(COALESCE(j.bars_total,0)),0)
              FROM jobs j WHERE j.run_id = r.id AND j.status != 'skipped') AS bar_progress
    FROM runs r ORDER BY r.id DESC`);
}

export function leaderboard(limit = 20): Row[] {
  return all(
    `SELECT * FROM v_strategy_summary WHERE strategy != 'random_entry' ORDER BY sharpe DESC LIMIT ?`,
    limit,
  );
}

export function initialBalance(run: Row | undefined): number {
  const cfg = parseJson<any>(run?.config_json, {});
  return Number(cfg?.engine?.initial_balance ?? 1_000_000);
}

export type StatRow = {
  broker: string;
  strategy: string;
  market: string;
  n: number;
  patterns: { market: string; seed: number; value: number | null; jobId: number; real: boolean }[];
  stats: Record<string, Summary | null>;
};

export const STAT_METRICS = ["pnl", "net_pips_per_year", "sharpe", "gross_sharpe", "max_dd", "trades_per_year", "win_rate"] as const;

export function runDetail(id: number) {
  const run = one(`SELECT * FROM runs WHERE id = ?`, id);
  if (!run) return null;
  const init = initialBalance(run);
  const paths = all(`SELECT * FROM paths WHERE run_id = ? ORDER BY market_label, seed`, id).map((p) => ({
    ...p,
    facts: parseJson<any>(p.facts_json, {}),
    facts_json: undefined,
    params_json: undefined,
  }));
  const jobs = all(
    `SELECT j.id, j.path_id, j.broker, j.strategy, j.strategy_label, j.timeframe, j.status, j.note,
            j.bars_total, j.bars_done, j.sim_t, j.balance, j.equity, j.n_trades, j.updated_at,
            j.pnl, j.sharpe, j.gross_sharpe, j.max_dd, j.net_pips_per_year, j.gross_pips_per_year,
            j.cost_pips_per_year, j.trades_per_year, j.win_rate, j.profit_factor, j.ruined, j.hunted,
            p.market_label, p.market, p.seed, p.detail
     FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ?
     ORDER BY j.broker, j.strategy_label, p.market_label, p.seed`,
    id,
  ).map((j): Row => ({ ...j, pnl_live: j.status === "done" ? j.pnl : j.equity != null ? j.equity - init : null }));

  // (ブローカー, 戦略, 市場) ごとと、市場をまとめた ALL
  const groups = new Map<string, Row[]>();
  for (const j of jobs) {
    if (j.status !== "done") continue;
    for (const m of [j.market_label, "ALL"]) {
      const k = `${j.broker}\u0000${j.strategy_label}\u0000${m}`;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(j);
    }
  }
  const stats: StatRow[] = [...groups.entries()].map(([k, rows]) => {
    const [broker, strategy, market] = k.split("\u0000");
    const st: Record<string, Summary | null> = {};
    for (const m of STAT_METRICS) st[m] = summarize(rows.map((r) => r[m]));
    return {
      broker,
      strategy,
      market,
      n: rows.length,
      patterns: rows.map((r) => ({
        market: r.market_label,
        seed: r.seed,
        value: r.pnl,
        jobId: r.id,
        real: r.market === "real",
      })),
      stats: st,
    };
  });
  stats.sort((a, b) =>
    a.broker.localeCompare(b.broker) || a.strategy.localeCompare(b.strategy) ||
    (a.market === "ALL" ? 1 : 0) - (b.market === "ALL" ? 1 : 0) || a.market.localeCompare(b.market),
  );
  const finished = jobs.filter((j) => DONE.includes(j.status)).length;
  return {
    run: { ...run, config: parseJson(run.config_json, {}), config_json: undefined },
    initialBalance: init,
    paths,
    jobs,
    stats,
    progress: { finished, total: jobs.length },
  };
}

type Candle = { time: number; open: number; high: number; low: number; close: number };

function aggregate(rows: Row[], tfMin: number): Candle[] {
  const size = tfMin * 60;
  const off = 7200; // 22:00 UTC で区切る(Python 側と同じ)
  const out: Candle[] = [];
  for (const r of rows) {
    const b = Math.floor((r.t + off) / size) * size - off;
    const last = out[out.length - 1];
    if (last && last.time === b) {
      last.high = Math.max(last.high, r.h);
      last.low = Math.min(last.low, r.l);
      last.close = r.c;
    } else {
      out.push({ time: b, open: r.o, high: r.h, low: r.l, close: r.c });
    }
  }
  return out;
}

export function jobDetail(id: number, q: { candleFrom: number; eqFrom: number; tradeFrom: number; tf: number | null }) {
  const job = one(
    `SELECT j.*, p.market_label, p.market, p.seed, p.detail, p.candle_tf, p.t_start, p.t_end, p.realism_score, p.data_key,
            r.name AS run_name, r.config_json
     FROM jobs j JOIN paths p ON p.id = j.path_id JOIN runs r ON r.id = j.run_id WHERE j.id = ?`,
    id,
  );
  if (!job) return null;
  const init = initialBalance(job);
  const candleTf = Number(job.candle_tf ?? 60);
  const key = job.data_key ?? "";
  const nCandles = one(`SELECT COUNT(*) AS n FROM candles WHERE data_key = ?`, key)?.n ?? 0;
  // 足が多いときは既定で粗くする(画面が重くならないように)
  const autoTf = nCandles > 40000 ? 1440 : nCandles > 12000 ? 240 : candleTf;
  const tf = Math.max(candleTf, q.tf ?? autoTf);
  const until = job.status === "done" ? Number.MAX_SAFE_INTEGER : Number(job.sim_t ?? 0);
  const raw = all(
    `SELECT t, o, h, l, c FROM candles WHERE data_key = ? AND t >= ? AND t <= ? ORDER BY t`,
    key,
    q.candleFrom,
    until,
  );
  const equity = all(
    `SELECT t, balance, equity FROM equity WHERE job_id = ? AND t >= ? ORDER BY t`,
    id,
    q.eqFrom,
  );
  const trades = all(`SELECT * FROM trades WHERE job_id = ? AND seq >= ? ORDER BY seq`, id, q.tradeFrom);
  const events = all(
    `SELECT t, kind, direction, pips FROM market_events
     WHERE data_key = ? AND t >= ? AND t <= ? AND kind != 'regime' ORDER BY t`,
    key,
    q.candleFrom,
    until,
  );
  return {
    job: {
      ...job,
      config_json: undefined,
      metrics: parseJson(job.metrics_json, null),
      metrics_json: undefined,
      params: parseJson(job.params_json, {}),
      params_json: undefined,
      schedule: parseJson(job.schedule_json, null),
      schedule_json: undefined,
      pnl_live: job.status === "done" ? job.pnl : job.equity != null ? job.equity - init : null,
    },
    initialBalance: init,
    tf,
    candleTf,
    candles: aggregate(raw, tf),
    equity,
    trades,
    events,
  };
}

export function casesOverview() {
  const has = one(`SELECT name FROM sqlite_master WHERE type='table' AND name='case_defs'`);
  if (!has) return { behaviors: [], tests: [] };
  const defs = all(`SELECT * FROM case_defs ORDER BY id`).map((d): Row => ({ ...d, def: parseJson(d.definition_json, {}), definition_json: undefined }));
  const runs = all(`SELECT c.*, r.status AS run_status, r.elapsed_sec FROM case_runs c LEFT JOIN runs r ON r.id = c.run_id ORDER BY c.id DESC`).map((r): Row => ({
    ...r,
    summary: parseJson(r.summary_json, null),
    summary_json: undefined,
  }));
  const tests = defs.filter((d) => d.kind === "test").map((t) => ({
    ...t,
    runs: runs.filter((r) => r.test_id === t.id),
    current: runs.find((r) => r.test_id === t.id)?.test_hash === t.hash,
  }));
  return { behaviors: defs.filter((d) => d.kind === "behavior"), tests };
}
