"use client";
import { useEffect, useRef } from "react";
import { ColorType, createChart, LineSeries, LineStyle, type UTCTimestamp } from "lightweight-charts";
import type { FanGroup } from "@/lib/types";

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888888";
}

/** "#rrggbb" に透明度を付ける(チャートのライブラリは color-mix を読めない)。 */
function alpha(hex: string, a: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return hex;
  return `rgba(${parseInt(m[1], 16)}, ${parseInt(m[2], 16)}, ${parseInt(m[3], 16)}, ${a})`;
}

/** パターンごとの損益の推移を重ねる。太線 = 中央値、点線 = 10/90 パーセンタイル、橙 = 実データ、破線 = 方向でたらめ版の中央値。 */
export default function FanChart({ group, height = 380 }: { group: FanGroup; height?: number }) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!el.current) return;
    const chart = createChart(el.current, {
      height,
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: css("--panel") }, textColor: css("--muted"), fontSize: 11 },
      grid: { vertLines: { color: css("--border") }, horzLines: { color: css("--border") } },
      rightPriceScale: { borderColor: css("--border") },
      timeScale: { borderColor: css("--border") },
      localization: { priceFormatter: (v: number) => `${Math.round(v / 10000)}万` },
    });
    const pts = (p: { t: number; v: number }[]) => p.map((x) => ({ time: x.t as UTCTimestamp, value: x.v }));
    const line = (color: string, width: 1 | 2 | 3 | 4, style: LineStyle = LineStyle.Solid) =>
      chart.addSeries(LineSeries, { color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

    const dot = css("--dot");
    for (const l of group.lines.filter((x) => !x.real)) line(alpha(dot, 0.35), 1).setData(pts(l.points));
    if (group.band.length) {
      line(css("--muted"), 1, LineStyle.Dotted).setData(group.band.map((b) => ({ time: b.t as UTCTimestamp, value: b.p10 })));
      line(css("--muted"), 1, LineStyle.Dotted).setData(group.band.map((b) => ({ time: b.t as UTCTimestamp, value: b.p90 })));
    }
    if (group.nullMedian.length) line(css("--muted"), 2, LineStyle.Dashed).setData(pts(group.nullMedian));
    if (group.band.length) {
      const med = line(css("--accent"), 3);
      med.setData(group.band.map((b) => ({ time: b.t as UTCTimestamp, value: b.median })));
      med.createPriceLine({ price: 0, color: css("--text"), lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: false, title: "" });
    }
    for (const l of group.lines.filter((x) => x.real)) line(css("--real"), 3).setData(pts(l.points));
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [group, height]);
  return <div ref={el} className="chart" style={{ height }} />;
}
