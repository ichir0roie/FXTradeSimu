import { all, one, parseJson, type Row } from "../db";
import type { Candle, EqPoint, JobResponse, MarketEvent, Trade } from "../types";
import { initialBalance, livePnl } from "./common";

const BAR_OFFSET_SEC = 7200; // 足の区切りを 22:00 UTC に揃える(Python の BAR_OFFSET_MINUTES と同じ)

/** 保存した足を tfMin 分足にまとめる。 */
function aggregate(rows: Row[], tfMin: number): Candle[] {
  const size = tfMin * 60;
  const out: Candle[] = [];
  for (const r of rows) {
    const time = Math.floor((r.t + BAR_OFFSET_SEC) / size) * size - BAR_OFFSET_SEC;
    const last = out[out.length - 1];
    if (last && last.time === time) {
      last.high = Math.max(last.high, r.h);
      last.low = Math.min(last.low, r.l);
      last.close = r.c;
    } else {
      out.push({ time, open: r.o, high: r.h, low: r.l, close: r.c });
    }
  }
  return out;
}

/** 足が多いときは既定で粗くする(画面が重くならないように)。 */
function defaultTimeframe(nCandles: number, candleTf: number): number {
  if (nCandles > 40000) return 1440;
  if (nCandles > 12000) return 240;
  return candleTf;
}

export type JobCursor = { candleFrom: number; eqFrom: number; tradeFrom: number; tf: number | null };

/** ジョブの情報と、cursor 以降の差分(足・資産曲線・取引・悪意イベント)。実行中は sim_t までしか見せない。 */
export function jobDetail(id: number, q: JobCursor): JobResponse | null {
  const job = one(
    `SELECT j.*, p.market_label, p.market, p.seed, p.detail, p.candle_tf, p.t_start, p.data_key,
            r.name AS run_name, r.config_json
     FROM jobs j JOIN paths p ON p.id = j.path_id JOIN runs r ON r.id = j.run_id WHERE j.id = ?`,
    id,
  );
  if (!job) return null;
  const init = initialBalance(job);
  const candleTf = Number(job.candle_tf ?? 60);
  const key = job.data_key ?? "";
  const nCandles = Number(one(`SELECT COUNT(*) AS n FROM candles WHERE data_key = ?`, key)?.n ?? 0);
  const tf = Math.max(candleTf, q.tf ?? defaultTimeframe(nCandles, candleTf));
  const until = job.status === "done" ? Number.MAX_SAFE_INTEGER : Number(job.sim_t ?? 0);

  const candles = aggregate(
    all(`SELECT t, o, h, l, c FROM candles WHERE data_key = ? AND t >= ? AND t <= ? ORDER BY t`, key, q.candleFrom, until),
    tf,
  );
  const equity = all(`SELECT t, balance, equity FROM equity WHERE job_id = ? AND t >= ? ORDER BY t`, id, q.eqFrom);
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
      id: job.id,
      run_id: job.run_id,
      run_name: job.run_name,
      strategy_label: job.strategy_label,
      market_label: job.market_label,
      seed: job.seed,
      broker: job.broker,
      status: job.status,
      timeframe: job.timeframe,
      bars_total: job.bars_total,
      bars_done: job.bars_done,
      sim_t: job.sim_t,
      t_start: job.t_start,
      balance: job.balance,
      equity: job.equity,
      n_trades: job.n_trades,
      pnl: job.pnl,
      pnl_live: livePnl(job, init),
      params: parseJson(job.params_json, {}),
      schedule: parseJson(job.schedule_json, null),
      metrics: parseJson(job.metrics_json, null),
    },
    initialBalance: init,
    tf,
    candleTf,
    candles,
    equity: equity as EqPoint[],
    trades: trades as Trade[],
    events: events as MarketEvent[],
  };
}
