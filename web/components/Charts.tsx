"use client";
import { useEffect, useRef } from "react";
import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, LineSeries,
  type IChartApi, type ISeriesApi, type SeriesMarker, type Time, type UTCTimestamp,
} from "lightweight-charts";

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type EqPoint = { t: number; balance: number; equity: number };
export type Trade = { seq: number; side: number; entry_t: number; exit_t: number; pnl: number; reason: string; pips: number };
export type MarketEvent = { t: number; kind: string; direction: number | null; pips: number | null };

const OFF = 7200; // 足の区切り = 22:00 UTC
const bucket = (t: number, tfMin: number) => Math.floor((t + OFF) / (tfMin * 60)) * tfMin * 60 - OFF;

function css(name: string): string {
  if (typeof window === "undefined") return "#888";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function baseOptions(height: number) {
  return {
    height,
    autoSize: true,
    layout: { background: { type: ColorType.Solid, color: css("--panel") }, textColor: css("--muted"), fontSize: 11 },
    grid: { vertLines: { color: css("--border") }, horzLines: { color: css("--border") } },
    rightPriceScale: { borderColor: css("--border") },
    timeScale: { borderColor: css("--border"), timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 },
  };
}

const EVENT_STYLE: Record<string, { color: string; text: string; label: string }> = {
  stop_hunt: { color: "#e07a10", text: "狩", label: "ストップ狩り(戻り)" },
  stop_cascade: { color: "#b8336a", text: "連", label: "ストップの連鎖" },
  false_breakout: { color: "#8a5cd6", text: "騙", label: "ダマシ" },
  breakout_run: { color: "#2a9d8f", text: "走", label: "抜けて走る" },
  news: { color: "#6b7280", text: "N", label: "指標" },
  weekend_gap: { color: "#6b7280", text: "窓", label: "週明けの窓" },
};
export const EVENT_LEGEND = EVENT_STYLE;

export function PriceChart({ candles, trades, events, tf, showTrades, showEvents, height = 420 }: {
  candles: Candle[]; trades: Trade[]; events: MarketEvent[]; tf: number; showTrades: boolean; showEvents: boolean; height?: number;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const markers = useRef<ReturnType<typeof createSeriesMarkers<Time>> | null>(null);
  const rendered = useRef<{ n: number; first: number; tf: number }>({ n: 0, first: -1, tf: 0 });

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, baseOptions(height));
    const s = c.addSeries(CandlestickSeries, {
      upColor: css("--up"), downColor: css("--down"), borderVisible: false,
      wickUpColor: css("--up"), wickDownColor: css("--down"),
    });
    chart.current = c;
    series.current = s;
    markers.current = createSeriesMarkers(s, []);
    (window as any).__fxPriceChart = c;
    return () => {
      c.remove();
      chart.current = null;
      series.current = null;
      rendered.current = { n: 0, first: -1, tf: 0 };
    };
  }, [height]);

  useEffect(() => {
    const s = series.current;
    if (!s || !candles.length) return;
    const r = rendered.current;
    const data = candles.map((c) => ({ ...c, time: c.time as UTCTimestamp }));
    if (r.n === 0 || r.first !== candles[0].time || r.tf !== tf || candles.length < r.n || candles.length - r.n > 500) {
      s.setData(data);
      const n = data.length;
      chart.current?.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 200), to: n + 3 });
    } else {
      for (let i = Math.max(0, r.n - 1); i < data.length; i++) s.update(data[i]);
    }
    rendered.current = { n: candles.length, first: candles[0].time, tf };
  }, [candles, tf]);

  useEffect(() => {
    if (!markers.current || !candles.length) return;
    const lastT = candles[candles.length - 1].time;
    const m: SeriesMarker<Time>[] = [];
    if (showTrades) {
      for (const t of trades.slice(-1500)) {
        const et = bucket(t.entry_t, tf);
        if (et <= lastT)
          m.push({
            time: et as UTCTimestamp,
            position: t.side > 0 ? "belowBar" : "aboveBar",
            shape: t.side > 0 ? "arrowUp" : "arrowDown",
            color: t.side > 0 ? css("--up") : css("--down"),
            text: t.side > 0 ? "買" : "売",
          });
        const xt = bucket(t.exit_t, tf);
        if (xt <= lastT)
          m.push({
            time: xt as UTCTimestamp,
            position: "inBar",
            shape: "circle",
            color: t.pnl >= 0 ? css("--pos") : css("--neg"),
            text: t.reason === "hunted" ? "狩" : t.reason === "sl" ? "損" : t.reason === "tp" ? "利" : "",
          });
      }
    }
    if (showEvents) {
      for (const e of events.slice(-1500)) {
        const st = EVENT_STYLE[e.kind];
        if (!st) continue;
        const et = bucket(e.t, tf);
        if (et > lastT) continue;
        m.push({
          time: et as UTCTimestamp,
          position: (e.direction ?? 1) > 0 ? "aboveBar" : "belowBar",
          shape: "square",
          color: st.color,
          text: st.text,
          size: 0.6,
        });
      }
    }
    m.sort((a, b) => (a.time as number) - (b.time as number));
    markers.current.setMarkers(m);
  }, [trades, events, candles, tf, showTrades, showEvents]);

  return <div ref={el} className="chart" style={{ height }} />;
}

export function EquityChart({ equity, initial, height = 200 }: { equity: EqPoint[]; initial: number; height?: number }) {
  const el = useRef<HTMLDivElement>(null);
  const eq = useRef<ISeriesApi<"Line"> | null>(null);
  const bal = useRef<ISeriesApi<"Line"> | null>(null);
  const chart = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, baseOptions(height));
    bal.current = c.addSeries(LineSeries, { color: css("--muted"), lineWidth: 1, lineStyle: 2, priceLineVisible: false, title: "残高" });
    eq.current = c.addSeries(LineSeries, { color: css("--accent"), lineWidth: 2, priceLineVisible: false, title: "有効証拠金" });
    eq.current.createPriceLine({ price: initial, color: css("--border"), lineWidth: 1, lineStyle: 1, axisLabelVisible: false, title: "" });
    chart.current = c;
    // 価格チャートと表示期間をそろえる
    const pc: IChartApi | undefined = (window as any).__fxPriceChart;
    const sync = (r: any) => {
      if (!r) return;
      try { c.timeScale().setVisibleRange(r); } catch { /* 範囲外は無視 */ }
    };
    pc?.timeScale().subscribeVisibleTimeRangeChange(sync);
    return () => {
      try { pc?.timeScale().unsubscribeVisibleTimeRangeChange(sync); } catch { /* removed */ }
      c.remove();
    };
  }, [height, initial]);

  useEffect(() => {
    if (!eq.current || !bal.current) return;
    eq.current.setData(equity.map((p) => ({ time: p.t as UTCTimestamp, value: p.equity })));
    bal.current.setData(equity.map((p) => ({ time: p.t as UTCTimestamp, value: p.balance })));
    const pc: IChartApi | undefined = (window as any).__fxPriceChart;
    const r = pc?.timeScale().getVisibleRange();
    if (r) {
      try { chart.current?.timeScale().setVisibleRange(r); } catch { chart.current?.timeScale().fitContent(); }
    } else {
      chart.current?.timeScale().fitContent();
    }
  }, [equity]);

  return <div ref={el} className="chart" style={{ height }} />;
}
