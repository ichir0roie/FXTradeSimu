"use client";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import type { IChartApi } from "lightweight-charts";
import { date, fmt, hoursLabel, neg, outsideLabel, pct, tfLabel, tone, yen } from "@/lib/format";
import { useJobStream } from "@/lib/useJobStream";
import type { CostBreakdown, JobInfo, JobMetrics, Trade } from "@/lib/types";
import { EVENT_STYLE } from "./Charts";
import { Progress, Status } from "./Progress";

const PriceChart = dynamic(() => import("./Charts").then((m) => m.PriceChart), { ssr: false });
const EquityChart = dynamic(() => import("./Charts").then((m) => m.EquityChart), { ssr: false });

const TIMEFRAMES = [60, 240, 1440];

export default function JobView({ id }: { id: number }) {
  const [tf, setTf] = useState<number | null>(null);
  const st = useJobStream(id, tf);
  const [showTrades, setShowTrades] = useState(true);
  // 悪意イベントは数が多いので、既定では H1 以下の足のときだけ表示する(ボタンで切り替え)
  const [eventsChoice, setEventsChoice] = useState<boolean | null>(null);
  const [priceChart, setPriceChart] = useState<IChartApi | null>(null);

  const tradeStats = useMemo(() => {
    const t = st?.trades ?? [];
    return {
      winRate: t.length ? t.filter((x) => x.pnl > 0).length / t.length : null,
      hunted: t.filter((x) => x.reason === "hunted").length,
    };
  }, [st?.trades]);

  if (!st) return <div className="empty">読み込み中…</div>;
  const j = st.job;
  const showEvents = eventsChoice ?? st.tf <= 60;

  return (
    <>
      <Header job={j} />

      <div className="tiles">
        <Tile k="シミュレーション日時" v={date(j.sim_t)} small />
        <Tile k="経過" v={j.sim_t && j.t_start ? `${fmt((j.sim_t - j.t_start) / 86400, 0)} 日` : "–"} />
        <Tile k="取引回数" v={String(j.n_trades ?? 0)} />
        <Tile k="残高(円)" v={fmt(j.balance, 0)} />
        <Tile k="有効証拠金(円)" v={fmt(j.equity, 0)} />
        <Tile k="損益(円)" v={yen(j.pnl_live)} cls={tone(j.pnl_live)} />
        <Tile k="勝率" v={pct(tradeStats.winRate ?? j.metrics?.win_rate)} />
        <Tile k="ブローカーに狩られた" v={String(j.metrics?.hunted ?? tradeStats.hunted)} />
      </div>

      {st.candles.length > 0 ? (
        <>
          <h2>価格</h2>
          <div className="controls">
            {TIMEFRAMES.filter((v) => v >= st.candleTf).map((v) => (
              <button key={v} className={st.tf === v ? "on" : ""} onClick={() => setTf(v)}>{tfLabel(v)}</button>
            ))}
            <span style={{ width: 12 }} />
            <button className={showTrades ? "on" : ""} onClick={() => setShowTrades(!showTrades)}>売買</button>
            <button className={showEvents ? "on" : ""} onClick={() => setEventsChoice(!showEvents)}>悪意イベント</button>
          </div>
          <PriceChart
            candles={st.candles} trades={st.trades} events={st.events} tf={st.tf}
            showTrades={showTrades} showEvents={showEvents} onChart={setPriceChart}
          />
          {showEvents && <EventLegend />}
        </>
      ) : (
        <div className="panel muted" style={{ marginTop: 14 }}>
          このパターンはローソク足を保存していません(ローソク足と取引は、設定の detail_seeds のシードだけ保存します)。資産曲線のみ表示します。
        </div>
      )}

      <h2>資産曲線</h2>
      <EquityChart equity={st.equity} initial={st.initial} syncWith={priceChart} />

      {j.metrics && <FinalMetrics pnl={j.pnl} m={j.metrics} />}
      {st.trades.length > 0 && <RecentTrades trades={st.trades} />}
    </>
  );
}

function Header({ job: j }: { job: JobInfo }) {
  const s = j.schedule;
  return (
    <>
      <div className="sub"><Link href={`/runs/${j.run_id}`}>← #{j.run_id} {j.run_name}</Link></div>
      <h1>
        {j.strategy_label} <span className="muted">on</span> {j.market_label}{" "}
        <span className="muted">seed {j.seed} / {j.broker}</span> <Status s={j.status} />
      </h1>
      <div className="sub">
        足 {tfLabel(j.timeframe)}
        {s && <> / 判断 {s.decision_every} 本ごと / 取引時間 {hoursLabel(s)} / 時間外 {outsideLabel(s.outside)}</>}
        {" "}/ パラメータ <code>{JSON.stringify(j.params)}</code>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "8px 0 12px" }}>
        <div style={{ flex: 1 }}><Progress value={j.bars_total ? (j.bars_done ?? 0) / j.bars_total : 0} /></div>
        <span className="num muted">{j.bars_done ?? 0}/{j.bars_total ?? 0} 本</span>
      </div>
    </>
  );
}

function Tile({ k, v, cls = "", small = false }: { k: string; v: string; cls?: string; small?: boolean }) {
  return (
    <div className="tile">
      <div className="k">{k}</div>
      <div className={`v ${cls}`} style={small ? { fontSize: 15 } : undefined}>{v}</div>
    </div>
  );
}

function EventLegend() {
  return (
    <div className="legend">
      {Object.entries(EVENT_STYLE).map(([k, v]) => (
        <span key={k}><i style={{ background: v.color, borderRadius: 2 }} />{v.text} = {v.label}</span>
      ))}
      <span>買/売 = 新規、利/損/狩 = 利確/損切り/ブローカーのストップ狩り</span>
    </div>
  );
}

function FinalMetrics({ pnl, m }: { pnl: number | null; m: JobMetrics }) {
  return (
    <>
      <h2>最終成績</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>損益(円)</th><th>CAGR</th><th>シャープ</th><th>素のシャープ</th><th>最大DD</th><th>PF</th><th>勝率</th>
              <th>取引/年</th><th>素のpip/年</th><th>コストpip/年</th><th>スワップ(円)</th><th className="l">決済理由</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className={`num ${tone(pnl)}`}>{yen(pnl)}</td>
              <td className="num">{pct(m.cagr)}</td>
              <td className={`num ${tone(m.sharpe)}`}>{fmt(m.sharpe)}</td>
              <td className={`num ${tone(m.gross_sharpe)}`}>{fmt(m.gross_sharpe)}</td>
              <td className="num">{pct(m.max_dd)}</td>
              <td className="num">{fmt(m.profit_factor)}</td>
              <td className="num">{pct(m.win_rate)}</td>
              <td className="num">{fmt(m.trades_per_year, 0)}</td>
              <td className={`num ${tone(m.gross_pips_per_year)}`}>{fmt(m.gross_pips_per_year, 0)}</td>
              <td className="num">{fmt(m.cost_pips_per_year, 0)}</td>
              <td className={`num ${tone(m.swap_total)}`}>{yen(m.swap_total)}</td>
              <td className="l">{Object.entries(m.exit_reasons ?? {}).map(([k, v]) => `${k}:${v}`).join(" ")}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {m.costs && <Costs c={m.costs} m={m} />}
    </>
  );
}

function Costs({ c, m }: { c: CostBreakdown; m: JobMetrics }) {
  const net = c.gross_pnl - c.total;
  return (
    <>
      <h2>費用の内訳</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>素の損益</th><th>スプレッド</th><th>スリッページ</th><th>手数料</th><th>スワップ</th>
              <th>費用の合計</th><th>取引の損益</th><th>ロスカット</th><th>拒否された注文</th><th>ゼロカット</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className={`num ${tone(c.gross_pnl)}`}>{yen(c.gross_pnl)}</td>
              <td className={`num ${tone(neg(c.spread))}`}>{yen(neg(c.spread))}</td>
              <td className={`num ${tone(neg(c.slippage))}`}>{yen(neg(c.slippage))}</td>
              <td className={`num ${tone(neg(c.commission))}`}>{yen(neg(c.commission))}</td>
              <td className={`num ${tone(c.swap)}`}>{yen(c.swap)}</td>
              <td className={`num ${tone(neg(c.total))}`}>{yen(neg(c.total))}</td>
              <td className={`num ${tone(net)}`}>{yen(net)}</td>
              <td className="num">{m.losscut ?? 0}</td>
              <td className="num">{m.rejected ?? 0}</td>
              <td className="num">{yen(m.zero_cut ?? 0)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  );
}

function RecentTrades({ trades }: { trades: Trade[] }) {
  return (
    <>
      <h2>直近の取引</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th className="l">売買</th><th className="l">建て</th><th className="l">決済</th><th>pip</th>
              <th>コストpip</th><th>損益(円)</th><th className="l">理由</th><th>保有(本)</th>
            </tr>
          </thead>
          <tbody>
            {trades.slice(-30).reverse().map((t) => (
              <tr key={t.seq}>
                <td className="num">{t.seq + 1}</td>
                <td className={`l ${t.side > 0 ? "pos" : "neg"}`}>{t.side > 0 ? "買" : "売"}</td>
                <td className="l num">{date(t.entry_t)} @ {fmt(t.entry_price, 3)}</td>
                <td className="l num">{date(t.exit_t)} @ {fmt(t.exit_price, 3)}</td>
                <td className={`num ${tone(t.pips)}`}>{fmt(t.pips, 1)}</td>
                <td className="num">{fmt(t.cost_pips, 1)}</td>
                <td className={`num ${tone(t.pnl)}`}>{yen(t.pnl)}</td>
                <td className="l">{t.reason}</td>
                <td className="num">{t.bars_held}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
