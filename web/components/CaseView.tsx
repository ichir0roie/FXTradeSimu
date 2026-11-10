"use client";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useState } from "react";
import { usePoll } from "@/lib/usePoll";
import { fmt, pct, tTone, tone, yen } from "@/lib/format";
import type { CaseDetail, VerdictLabel, YearRow } from "@/lib/types";
import { Progress } from "./Progress";

const FanChart = dynamic(() => import("./FanChart"), { ssr: false });

const VERDICT_CLASS: Record<VerdictLabel, string> = { 合格: "pass", 条件付き: "warn", 不合格: "fail" };

export default function CaseView({ id }: { id: string }) {
  const { data } = usePoll<CaseDetail & { error?: string }>(
    `/api/cases/${encodeURIComponent(id)}`,
    (d) => !d || d.runStatus === "running",
    2000,
  );
  const [groupKey, setGroupKey] = useState("bootstrap");
  if (!data) return <div className="empty">読み込み中…</div>;
  if (data.error) return <div className="empty">{id} は見つかりません</div>;
  const v = data.verdict;
  const group = data.fan.find((g) => g.key === groupKey) ?? data.fan[0];
  const t = data.test.def;
  const period = t.period?.years ? `${t.period.years} 年` : `${t.period?.start}〜${t.period?.end}`;

  return (
    <>
      <div className="sub"><Link href="/cases">← テストケース</Link></div>
      <h1>{data.test.id} {data.test.name}</h1>
      <div className="sub">
        動作ケース {t.behavior}{data.behavior && <>({data.behavior.name})</>} / 期間 {period}
        {data.runId && <> / <Link href={`/runs/${data.runId}`}>run #{data.runId} の詳細 →</Link></>}
      </div>
      {data.runStatus === "running" && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "10px 0" }}>
          <div style={{ flex: 1 }}><Progress value={data.progress.total ? data.progress.finished / data.progress.total : 0} /></div>
          <span className="num muted">{data.progress.finished}/{data.progress.total} ジョブ</span>
        </div>
      )}

      {v ? (
        <div className={`verdict ${VERDICT_CLASS[v.verdict]}`}>
          <div className="verdict-label">判定: {v.verdict}</div>
          <ul>
            {v.checks.map((c) => (
              <li key={c.key}>
                <span className={c.ok ? "pos" : "neg"}>{c.ok ? "✓" : "✗"}</span> <b>{c.key} {c.label}</b>
                <span className="muted"> — {c.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="panel muted" style={{ marginTop: 12 }}>
          このテストケースには判定がありません(定義に verify: true を付けるか、python -m fxsim verify で検証します)。
        </div>
      )}

      {v && (
        <>
          <h2>パターンの組ごとの損益(円)</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr><th className="l">組</th><th>n</th><th>平均</th><th>中央</th><th>合計</th><th>最小</th><th>最大</th><th>プラスの割合</th><th>t値</th><th>方向でたらめ版の平均</th></tr>
              </thead>
              <tbody>
                {Object.entries(v.groups).map(([k, g]) => (
                  <tr key={k}>
                    <td className="l">{g.label}</td>
                    <td className="num">{g.n}</td>
                    <td className={`num ${tone(g.mean)}`}>{yen(g.mean)}</td>
                    <td className={`num ${tone(g.median)}`}>{yen(g.median)}</td>
                    <td className={`num ${tone(g.total)}`}>{yen(g.total)}</td>
                    <td className={`num ${tone(g.min)}`}>{yen(g.min)}</td>
                    <td className={`num ${tone(g.max)}`}>{yen(g.max)}</td>
                    <td className="num">{pct(g.positive_ratio, 0)}</td>
                    <td className={`num ${tTone(g.t)}`}>{fmt(g.t)}</td>
                    <td className={`num ${tone(g.null_mean)}`}>{yen(g.null_mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {group && (
        <>
          <h2>資産曲線の束(損益の推移)</h2>
          <div className="controls">
            {data.fan.map((g) => (
              <button key={g.key} className={g.key === group.key ? "on" : ""} onClick={() => setGroupKey(g.key)}>
                {g.label}({g.lines.filter((l) => !l.real).length})
              </button>
            ))}
          </div>
          <div className="legend">
            <span><i style={{ background: "var(--accent)", borderRadius: 0, height: 4 }} />中央値</span>
            <span><i style={{ background: "var(--dot)" }} />ランダム化したパターン</span>
            <span><i style={{ background: "var(--real)" }} />実データ</span>
            <span>点線 = 10/90 パーセンタイル / 破線 = 方向でたらめ版の中央値 / 横線 = 損益 0</span>
          </div>
          <FanChart group={group} />
        </>
      )}

      {data.yearRows.length > 0 && <YearHeatmap years={data.years} rows={data.yearRows} />}
    </>
  );
}

function YearHeatmap({ years, rows }: { years: string[]; rows: YearRow[] }) {
  const maxAbs = Math.max(1, ...rows.flatMap((r) => Object.values(r.values).map(Math.abs)));
  const cell = (v: number | undefined) => {
    if (v === undefined) return { background: "transparent" };
    const a = Math.round(Math.min(1, Math.abs(v) / maxAbs) * 70) + 8;
    return { background: `color-mix(in srgb, var(${v >= 0 ? "--pos" : "--neg"}) ${a}%, transparent)` };
  };
  const medians = years.map((y) => {
    const v = rows.map((r) => r.values[y]).filter((x): x is number => x !== undefined).sort((a, b) => a - b);
    return v.length ? v[Math.floor((v.length - 1) / 2)] : undefined;
  });
  return (
    <>
      <h2>年ごとの損益(実データと並べ替えのパターン、万円)</h2>
      <div className="tablewrap">
        <table className="heat">
          <thead>
            <tr><th className="l">パターン</th>{years.map((y) => <th key={y}>{y}</th>)}<th>合計</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="l">{r.real ? <span className="badge real">実データ</span> : r.label}</td>
                {years.map((y) => <td key={y} className="num" style={cell(r.values[y])}>{fmt(r.values[y] !== undefined ? r.values[y] / 10000 : undefined, 1)}</td>)}
                <td className={`num ${tone(Object.values(r.values).reduce((a, b) => a + b, 0))}`}>
                  {fmt(Object.values(r.values).reduce((a, b) => a + b, 0) / 10000, 1)}
                </td>
              </tr>
            ))}
            <tr className="all">
              <td className="l">中央値</td>
              {medians.map((m, i) => <td key={years[i]} className="num" style={cell(m)}>{fmt(m !== undefined ? m / 10000 : undefined, 1)}</td>)}
              <td />
            </tr>
          </tbody>
        </table>
      </div>
      <div className="sub" style={{ marginTop: 4 }}>期間の端の年は数か月分だけのことがある(判定 C4 では半年未満の年を除く)。中央値は並べ替えのパターンと実データを合わせたもの。</div>
    </>
  );
}
