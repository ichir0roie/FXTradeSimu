"use client";
import Link from "next/link";
import { usePoll } from "@/lib/usePoll";
import { fmt, pct, yen, tone } from "@/lib/format";
import { Status } from "./Progress";

function hours(s: any): string {
  if (!s?.hours?.length) return "終日";
  return s.hours.map((h: string[]) => `${h[0]}–${h[1]}`).join(", ") + ` ${s.tz}`;
}

export default function CasesView() {
  const { data } = usePoll<any>("/api/cases", (d) => !!d?.tests?.some((t: any) => t.runs[0]?.status === "running"), 2000);
  if (!data) return <div className="empty">読み込み中…</div>;
  const behaviors = new Map<string, any>(data.behaviors.map((b: any) => [b.id, b]));
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
            {data.tests.map((t: any) => {
              const r = t.runs[0];
              const a = r?.summary?.all;
              const p = a?.pnl;
              const per = t.def.period?.years ? `${t.def.period.years} 年` : `${t.def.period?.start}〜${t.def.period?.end}`;
              return (
                <tr key={t.id}>
                  <td className="l">{t.id}</td>
                  <td className="l" style={{ whiteSpace: "normal", minWidth: 220 }}>{t.name}</td>
                  <td className="l">{t.def.behavior}</td>
                  <td className="l">{per}</td>
                  <td className="l" style={{ whiteSpace: "normal" }}>
                    {t.def.patterns.map((x: any) => `${x.label ?? x.market}×${x.seeds.length}`).join(", ")}
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
                  <td className={`num ${Math.abs(p?.t ?? 0) >= 2 ? tone(p?.t) : "muted"}`}>{fmt(p?.t)}</td>
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
            {data.tests.flatMap((t: any) => {
              const bm = t.runs[0]?.summary?.by_market ?? {};
              return Object.entries(bm).map(([m, v]: [string, any]) => (
                <tr key={`${t.id}-${m}`}>
                  <td className="l">{t.id}</td>
                  <td className="l">{m} {m === "real" && <span className="badge real">実データ</span>}</td>
                  <td className="num">{v.pnl?.n}</td>
                  <td className={`num ${tone(v.pnl?.mean)}`}>{yen(v.pnl?.mean)}</td>
                  <td className={`num ${tone(v.pnl?.median)}`}>{yen(v.pnl?.median)}</td>
                  <td className={`num ${tone(v.pnl?.total)}`}>{yen(v.pnl?.total)}</td>
                  <td className="num">{fmt(v.pnl?.std, 0)}</td>
                  <td className="num">{pct(v.pnl?.positive_ratio, 0)}</td>
                  <td className={`num ${Math.abs(v.pnl?.t ?? 0) >= 2 ? tone(v.pnl?.t) : "muted"}`}>{fmt(v.pnl?.t)}</td>
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
            {[...behaviors.values()].map((b: any) => {
              const d = b.def;
              const tf = d.timeframe ?? d.params?.timeframe;
              return (
                <tr key={b.id}>
                  <td className="l">{b.id}</td>
                  <td className="l" style={{ whiteSpace: "normal", minWidth: 200 }}>{b.name}</td>
                  <td className="l">{d.strategy}</td>
                  <td className="l muted" style={{ whiteSpace: "normal" }}><code>{JSON.stringify(d.params)}</code></td>
                  <td className="num">{tf ? (tf >= 1440 ? "D1" : tf >= 60 ? `H${tf / 60}` : `M${tf}`) : "既定"}</td>
                  <td className="num">{d.schedule?.decision_every} 本ごと</td>
                  <td className="l">{hours(d.schedule)}</td>
                  <td className="l">{d.schedule?.days?.join(" ")}</td>
                  <td className="l">{d.schedule?.outside === "close" ? "手仕舞い" : "持ち越し"}</td>
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
