"use client";
import { useEffect, useRef, useState } from "react";
import type { Candle, EqPoint, JobInfo, JobResponse, MarketEvent, Trade } from "./types";

export type JobStream = {
  job: JobInfo;
  initial: number;
  tf: number;
  candleTf: number;
  candles: Candle[];
  equity: EqPoint[];
  trades: Trade[];
  events: MarketEvent[];
};

/** cursor より前のものを残し、後ろは届いた差分で置き換える(最後の足は未確定なので取り直す)。 */
function merge<T>(prev: T[] | undefined, from: number, key: (x: T) => number, next: T[]): T[] {
  return [...(prev ?? []).filter((x) => key(x) < from), ...next];
}

/**
 * ジョブを 1 秒ごとに差分で取りに行く。実行中・待機中の間だけ続ける。
 * tf(足の分)を変えると最初から取り直す。
 */
export function useJobStream(id: number, tf: number | null): JobStream | null {
  const [state, setState] = useState<JobStream | null>(null);
  const cursor = useRef({ candleFrom: 0, eqFrom: 0, tradeFrom: 0 });

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    cursor.current = { candleFrom: 0, eqFrom: 0, tradeFrom: 0 };
    setState(null);

    const tick = async () => {
      const c = { ...cursor.current };
      const q = new URLSearchParams({
        candleFrom: String(c.candleFrom),
        eqFrom: String(c.eqFrom),
        tradeFrom: String(c.tradeFrom),
      });
      if (tf) q.set("tf", String(tf));
      try {
        const res = await fetch(`/api/jobs/${id}?${q}`, { cache: "no-store" });
        const d = (await res.json()) as JobResponse & { error?: string };
        if (stop || d.error) return;
        setState((prev) => ({
          job: d.job,
          initial: d.initialBalance,
          tf: d.tf,
          candleTf: d.candleTf,
          candles: merge(prev?.candles, c.candleFrom, (x) => x.time, d.candles),
          equity: merge(prev?.equity, c.eqFrom, (x) => x.t, d.equity),
          events: merge(prev?.events, c.candleFrom, (x) => x.t, d.events),
          trades: [...(prev?.trades ?? []), ...d.trades],
        }));
        cursor.current = {
          candleFrom: d.candles.at(-1)?.time ?? c.candleFrom,
          eqFrom: d.equity.at(-1)?.t ?? c.eqFrom,
          tradeFrom: c.tradeFrom + d.trades.length,
        };
        if (d.job.status === "running" || d.job.status === "queued") timer = setTimeout(tick, 1000);
      } catch {
        if (!stop) timer = setTimeout(tick, 3000);
      }
    };
    tick();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  }, [id, tf]);

  return state;
}
