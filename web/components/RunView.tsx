"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { usePoll } from "@/lib/usePoll";
import { fmt, pct, tfLabel, tone, tTone, yen } from "@/lib/format";
import { summarize } from "@/lib/stats";
import type { JobRow, MetricKey, PathRow, RunDetail } from "@/lib/types";
import { Progress, Status } from "./Progress";
import { StripPlot, type Pattern } from "./StripPlot";

type Fmt = (x: number | null | undefined) => string;

const METRICS: { key: MetricKey; label: string; f: Fmt; ratio: boolean }[] = [
  { key: "pnl", label: "損益(円)", f: yen, ratio: false },
  { key: "net_pips_per_year", label: "pip/年", f: (x) => fmt(x, 0), ratio: false },
  { key: "sharpe", label: "シャープ", f: (x) => fmt(x), ratio: false },
  { key: "gross_sharpe", label: "素のシャープ(コスト抜き)", f: (x) => fmt(x), ratio: false },
  { key: "max_dd", label: "最大DD", f: (x) => pct(x), ratio: true },
  { key: "trades_per_year", label: "取引/年", f: (x) => fmt(x, 0), ratio: false },
  { key: "win_rate", label: "勝率", f: (x) => pct(x), ratio: true },
];

const FACTS: [string, string, number][] = [
  ["annual_vol", "年率ボラ", 3],
  ["kurt_h1", "尖度H1", 1],
  ["acf_abs_h1_lag1", "|r|自己相関", 3],
  ["vr_h1_d1", "分散比H1→D1", 3],
  ["daily_range_pct", "日足値幅%", 2],
  ["spread_p99_pips", "スプレッドp99", 2],
  ["false_breakout_rate", "ダマシ率", 3],
  ["wick_rejection_rate", "ヒゲ戻り率", 3],
];

export default function RunView({ id }: { id: number }) {
  const { data } = usePoll<RunDetail & { error?: string }>(`/api/runs/${id}`, (d) => !d || d.run?.status === "running", 1500);
  if (!data) return <div className="empty">読み込み中…</div>;
  if (data.error) return <div className="empty">run #{id} は見つかりません</div>;
  const { run, progress } = data;
  return (
    <>
      <div className="sub"><Link href="/">← 実験一覧</Link></div>
      <h1>#{run.id} {run.name} <Status s={run.status} /></h1>
      <div className="sub">
        {run.kind} / 期間 {fmt(run.years, 1)} 年 / シード {run.seeds_json} / git {run.git_sha ?? "–"} / 初期資金{" "}
        {fmt(data.initialBalance, 0)} 円{run.proposal && <> / 提案 {run.proposal}</>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "10px 0" }}>
        <div style={{ flex: 1 }}><Progress value={progress.total ? progress.finished / progress.total : 0} /></div>
        <span className="num muted">{progress.finished}/{progress.total} ジョブ</span>
      </div>
      <MarketFacts paths={data.paths} />
      <StrategyStats data={data} />
      <JobTable jobs={data.jobs} />
    </>
  );
}

function MarketFacts({ paths }: { paths: PathRow[] }) {
  const markets = useMemo(() => {
    const g = new Map<string, PathRow[]>();
    for (const p of paths) g.set(p.market_label, [...(g.get(p.market_label) ?? []), p]);
    return [...g.entries()];
  }, [paths]);
  return (
    <>
      <h2>市場の性質(シード平均)</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="l">市場</th><th>パターン</th><th>現実らしさ</th><th>目安外</th>
              {FACTS.map(([k, l]) => <th key={k}>{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {markets.map(([label, ps]) => {
              const done = ps.filter((p) => p.status === "done" || p.status === "running");
              const mean = (f: (p: PathRow) => number | null | undefined) => summarize(done.map(f))?.mean;
              return (
                <tr key={label}>
                  <td className="l">{label} {ps.some((p) => p.market === "real") && <span className="badge real">実データ</span>}</td>
                  <td className="num">{ps.length}</td>
                  <td className="num">{fmt(mean((p) => p.realism_score), 1)}</td>
                  <td className="num">{fmt(mean((p) => p.n_fail), 1)}</td>
                  {FACTS.map(([k, , d]) => <td key={k} className="num">{fmt(mean((p) => p.facts?.[k]), d)}</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function StrategyStats({ data }: { data: RunDetail }) {
  const [metric, setMetric] = useState<MetricKey>("pnl");
  const [brokerChoice, setBroker] = useState<string | null>(null);
  const jobsById = useMemo(() => new Map(data.jobs.map((j) => [j.id, j])), [data.jobs]);
  const brokers = useMemo(() => [...new Set(data.jobs.map((j) => j.broker))], [data.jobs]);
  const m = METRICS.find((x) => x.key === metric)!;
  const broker = brokerChoice ?? brokers[0];

  const rows = data.groups
    .filter((g) => g.broker === broker)
    .map((g) => {
      const patterns: Pattern[] = g.jobIds.map((id) => {
        const j = jobsById.get(id)!;
        return { value: j[metric], real: j.market === "real", seed: j.seed, market: j.market_label };
      });
      return { ...g, patterns, st: summarize(patterns.map((p) => p.value)) };
    });
  const values = rows.flatMap((r) => r.patterns.map((p) => p.value)).filter((v): v is number => v != null);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);

  return (
    <>
      <h2>戦略の成績(パターンごとの統計)</h2>
      <div className="controls">
        {METRICS.map((x) => (
          <button key={x.key} className={x.key === metric ? "on" : ""} onClick={() => setMetric(x.key)}>{x.label}</button>
        ))}
        <span className="muted" style={{ marginLeft: 12 }}>ブローカー:</span>
        {brokers.map((x) => (
          <button key={x} className={x === broker ? "on" : ""} onClick={() => setBroker(x)}>{x}</button>
        ))}
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--dot)" }} />ランダム相場のパターン</span>
        <span><i style={{ background: "var(--real)", borderRadius: 0, transform: "rotate(45deg) scale(.8)" }} />実データ</span>
        <span>実線 = 平均 / 破線 = 中央値 / 太字の行 = 全市場の総合</span>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="l">戦略</th><th className="l">市場</th><th>n</th><th>平均</th><th>中央</th>
              {!m.ratio && <th>合計</th>}<th>標準偏差</th><th>最小</th><th>最大</th><th>プラスの割合</th>
              <th title="パターン単位の t 値。|t| ≥ 2 が有意の目安">t値</th><th className="l">分布</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.strategy}-${r.market}`} className={r.market === "ALL" ? "all" : ""}>
                <td className="l">{r.strategy}</td>
                <td className="l">{r.market === "ALL" ? "総合" : r.market}</td>
                <td className="num">{r.st?.n ?? 0}</td>
                <td className={`num ${tone(r.st?.mean)}`}>{m.f(r.st?.mean)}</td>
                <td className={`num ${tone(r.st?.median)}`}>{m.f(r.st?.median)}</td>
                {!m.ratio && <td className={`num ${tone(r.st?.total)}`}>{m.f(r.st?.total)}</td>}
                <td className="num">{m.f(r.st?.std)}</td>
                <td className={`num ${tone(r.st?.min)}`}>{m.f(r.st?.min)}</td>
                <td className={`num ${tone(r.st?.max)}`}>{m.f(r.st?.max)}</td>
                <td className="num">{pct(r.st?.positive_ratio, 0)}</td>
                <td className={`num ${tTone(r.st?.t)}`}>{fmt(r.st?.t)}</td>
                <td className="l"><StripPlot patterns={r.patterns} lo={lo} hi={hi} mean={r.st?.mean} median={r.st?.median} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function JobTable({ jobs: allJobs }: { jobs: JobRow[] }) {
  const [filter, setFilter] = useState<"detail" | "all">("detail");
  const [limit, setLimit] = useState(40);
  const jobs = allJobs.filter((j) => filter === "all" || j.detail || j.status === "running");
  return (
    <>
      <h2>ジョブ</h2>
      <div className="controls">
        <button className={filter === "detail" ? "on" : ""} onClick={() => setFilter("detail")}>詳細保存のみ</button>
        <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>すべて</button>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="l">市場</th><th>seed</th><th className="l">ブローカー</th><th className="l">戦略</th><th>足</th>
              <th className="l">状態</th><th className="l">進捗</th><th>取引</th><th>損益(円)</th><th>シャープ</th><th>最大DD</th><th className="l"></th>
            </tr>
          </thead>
          <tbody>
            {jobs.slice(0, limit).map((j) => (
              <tr key={j.id}>
                <td className="l">{j.market_label}</td>
                <td className="num">{j.seed}</td>
                <td className="l">{j.broker}</td>
                <td className="l">{j.strategy_label}</td>
                <td className="num">{tfLabel(j.timeframe)}</td>
                <td className="l"><Status s={j.status} /></td>
                <td className="l" style={{ minWidth: 110 }}>
                  {j.status === "skipped"
                    ? <span className="muted" title={j.note ?? ""}>{j.note}</span>
                    : <Progress value={j.bars_total ? (j.bars_done ?? 0) / j.bars_total : 0} />}
                </td>
                <td className="num">{j.n_trades ?? "–"}</td>
                <td className={`num ${tone(j.pnl_live)}`}>{yen(j.pnl_live)}</td>
                <td className={`num ${tone(j.sharpe)}`}>{fmt(j.sharpe)}</td>
                <td className="num">{pct(j.max_dd)}</td>
                <td className="l">{j.status !== "skipped" && <Link href={`/jobs/${j.id}`}>{j.detail ? "チャート" : "資産曲線"} →</Link>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {jobs.length > limit && (
        <div className="controls" style={{ justifyContent: "center", marginTop: 10 }}>
          <span className="muted">{limit} / {jobs.length} 件を表示</span>
          <button onClick={() => setLimit(jobs.length)}>すべて表示</button>
        </div>
      )}
    </>
  );
}
