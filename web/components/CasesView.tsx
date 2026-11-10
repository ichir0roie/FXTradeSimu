"use client";
import Link from "next/link";
import { usePoll } from "@/lib/usePoll";
import { fmt, hoursLabel, outsideLabel, pct, tfLabel, tone, tTone, yen } from "@/lib/format";
import type { CasesResponse, Summary } from "@/lib/types";
import { Status } from "./Progress";

export default function CasesView() {
  const { data } = usePoll<CasesResponse>("/api/cases", (d) => !!d?.tests.some((t) => t.runs[0]?.status === "running"), 2000);
  if (!data) return <div className="empty">読み込み中…</div>;
  const behaviors = data.behaviors;
  return (
    <>
      <h1>テストケース</h1>
      <div className="sub">
        定義は <code>cases/behaviors</code>(動作ケース)と <code>cases/tests</code>(テストケース)。
        <code>python -m fxsim cases run --all</code> で機械的に実行され、ここに記録されます。
      </div>

      <h2>テストケースと最新の結果(損益は円、全パターンの総合)</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="l">ID</th><th className="l">名前</th><th className="l">動作</th><th className="l">期間</th><th className="l">パターン</th>
              <th className="l">状態</th><th>n</th><th>平均</th><th>中央</th><th>合計</th><th>最小</th><th>最大</th><th>プラスの割合</th><th>t値</th><th>シャープ平均</th><th className="l">run</th>
            </tr>
          </thead>
          <tbody>
            {data.tests.map((t) => {
              const r = t.runs[0];
              const a = r?.summary?.all;
              const p: Summary | undefined = a?.pnl;
              const per = t.def.period?.years ? `${t.def.period.years} 年` : `${t.def.period?.start}〜${t.def.period?.end}`;
              return (
                <tr key={t.id}>
                  <td className="l">{t.id}</td>
                  <td className="l" style={{ whiteSpace: "normal", minWidth: 220 }}>{t.name}</td>
                  <td className="l">{t.def.behavior}</td>
                  <td className="l">{per}</td>
                  <td className="l" style={{ whiteSpace: "normal" }}>
                    {(t.def.patterns ?? []).map((x) => `${x.label ?? x.market}×${x.seeds.length}`).join(", ")}
                  </td>
                  <td className="l">
                    {r ? <Status s={r.status} /> : <span className="badge">未実行</span>}
                    {r && !t.current && <span className="badge" style={{ marginLeft: 4 }} title="定義が変わっています">要再実行</span>}
                  </td>
                  <td className="num">{p?.n ?? "–"}</td>
                  <td className={`num ${tone(p?.mean)}`}>{yen(p?.mean)}</td>
                  <td className={`num ${tone(p?.median)}`}>{yen(p?.median)}</td>
                  <td className={`num ${tone(p?.total)}`}>{yen(p?.total)}</td>
                  <td className={`num ${tone(p?.min)}`}>{yen(p?.min)}</td>
                  <td className={`num ${tone(p?.max)}`}>{yen(p?.max)}</td>
                  <td className="num">{pct(p?.positive_ratio, 0)}</td>
                  <td className={`num ${tTone(p?.t)}`}>{fmt(p?.t)}</td>
                  <td className={`num ${tone(a?.sharpe?.mean)}`}>{fmt(a?.sharpe?.mean)}</td>
                  <td className="l">{r?.run_id ? <Link href={`/runs/${r.run_id}`}>#{r.run_id} →</Link> : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h2>パターン別(最新の実行)</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th className="l">テスト</th><th className="l">相場パターン</th><th>n</th><th>平均</th><th>中央</th><th>合計</th><th>標準偏差</th><th>プラスの割合</th><th>t値</th><th>シャープ平均</th><th>最大DD平均</th></tr>
          </thead>
          <tbody>
            {data.tests.flatMap((t) => {
              const bm = t.runs[0]?.summary?.by_market ?? {};
              return Object.entries(bm).map(([m, v]) => (
                <tr key={`${t.id}-${m}`}>
                  <td className="l">{t.id}</td>
                  <td className="l">{m} {m === "real" && <span className="badge real">実データ</span>}</td>
                  <td className="num">{v.pnl?.n}</td>
                  <td className={`num ${tone(v.pnl?.mean)}`}>{yen(v.pnl?.mean)}</td>
                  <td className={`num ${tone(v.pnl?.median)}`}>{yen(v.pnl?.median)}</td>
                  <td className={`num ${tone(v.pnl?.total)}`}>{yen(v.pnl?.total)}</td>
                  <td className="num">{fmt(v.pnl?.std, 0)}</td>
                  <td className="num">{pct(v.pnl?.positive_ratio, 0)}</td>
                  <td className={`num ${tTone(v.pnl?.t)}`}>{fmt(v.pnl?.t)}</td>
                  <td className={`num ${tone(v.sharpe?.mean)}`}>{fmt(v.sharpe?.mean)}</td>
                  <td className="num">{pct(v.max_dd?.mean)}</td>
                </tr>
              ));
            })}
          </tbody>
        </table>
      </div>

      <h2>動作ケース</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th className="l">ID</th><th className="l">名前</th><th className="l">戦略</th><th className="l">パラメータ</th><th>足</th><th>判断の間隔</th><th className="l">取引時間</th><th className="l">曜日</th><th className="l">時間外</th><th className="l">ブローカー</th></tr>
          </thead>
          <tbody>
            {behaviors.map((b) => {
              const d = b.def;
              const tf = d.timeframe ?? (d.params?.timeframe as number | undefined);
              return (
                <tr key={b.id}>
                  <td className="l">{b.id}</td>
                  <td className="l" style={{ whiteSpace: "normal", minWidth: 200 }}>{b.name}</td>
                  <td className="l">{d.strategy}</td>
                  <td className="l muted" style={{ whiteSpace: "normal" }}><code>{JSON.stringify(d.params)}</code></td>
                  <td className="num">{tfLabel(tf)}</td>
                  <td className="num">{d.schedule?.decision_every} 本ごと</td>
                  <td className="l">{hoursLabel(d.schedule)}</td>
                  <td className="l">{d.schedule?.days?.join(" ")}</td>
                  <td className="l">{outsideLabel(d.schedule?.outside)}</td>
                  <td className="l">{typeof d.broker === "string" ? d.broker : d.broker?.preset}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
