"use client";
import Link from "next/link";
import { usePoll } from "@/lib/usePoll";
import { fmt, pct, yen, tone } from "@/lib/format";
import type { RunsResponse } from "@/lib/types";
import { Progress, Status } from "./Progress";


export default function RunsView() {
  const { data } = usePoll<RunsResponse>("/api/runs", () => true, 2000);
  if (!data) return <div className="empty">読み込み中…</div>;
  if (data.missing)
    return (
      <div className="empty">
        データベースがまだありません: <code>{data.db}</code>
        <br />
        リポジトリ直下で <code>python -m fxsim demo</code> などを実行すると作られます。
      </div>
    );
  return (
    <>
      <h1>実験</h1>
      <div className="sub">シミュレーション中の実験は 2 秒ごとに更新されます。</div>
      <div className="tablewrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>#</th><th className="l">名前</th><th className="l">種類</th><th className="l">状態</th>
              <th className="l">進捗</th><th>ジョブ</th><th>経路</th><th>期間</th><th className="l">作成</th><th>所要</th><th className="l">git</th>
            </tr>
          </thead>
          <tbody>
            {data.runs.map((r) => (
              <tr key={r.id}>
                <td className="num">{r.id}</td>
                <td className="l"><Link href={`/runs/${r.id}`}>{r.name}</Link>{r.proposal && <span className="badge" style={{ marginLeft: 6 }}>{r.proposal}</span>}</td>
                <td className="l muted">{r.kind}</td>
                <td className="l"><Status s={r.status} /></td>
                <td className="l" style={{ minWidth: 140 }}><Progress value={r.status === "done" ? 1 : r.bar_progress} /></td>
                <td className="num">{r.n_finished}/{r.n_jobs}</td>
                <td className="num">{r.n_paths}</td>
                <td className="num">{fmt(r.years, 1)}年</td>
                <td className="l muted">{r.created_at?.replace("T", " ").replace("Z", "")}</td>
                <td className="num">{r.elapsed_sec ? `${fmt(r.elapsed_sec, 0)}s` : "–"}</td>
                <td className="l muted num">{r.git_sha ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>戦略ランキング(全実験・シャープ順、random_entry を除く)</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>run</th><th className="l">市場</th><th className="l">ブローカー</th><th className="l">戦略</th><th>n</th>
              <th>シャープ</th><th>素のシャープ</th><th>損益 平均</th><th>損益 合計</th><th>pip/年</th><th>最大DD</th>
            </tr>
          </thead>
          <tbody>
            {data.leaderboard.map((r, i) => (
              <tr key={i}>
                <td className="num"><Link href={`/runs/${r.run_id}`}>{r.run_id}</Link></td>
                <td className="l">{r.market_label}</td>
                <td className="l">{r.broker}</td>
                <td className="l">{r.strategy_label}</td>
                <td className="num">{r.n_seeds}</td>
                <td className={`num ${tone(r.sharpe)}`}>{fmt(r.sharpe)}</td>
                <td className={`num ${tone(r.gross_sharpe)}`}>{fmt(r.gross_sharpe)}</td>
                <td className={`num ${tone(r.pnl_mean)}`}>{yen(r.pnl_mean)}</td>
                <td className={`num ${tone(r.pnl_total)}`}>{yen(r.pnl_total)}</td>
                <td className={`num ${tone(r.net_pips_per_year)}`}>{fmt(r.net_pips_per_year, 0)}</td>
                <td className="num">{pct(r.max_dd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
