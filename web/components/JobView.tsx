"use client";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { date, fmt, pct, yen, tone } from "@/lib/format";
import { Progress, Status } from "./Progress";
import type { Candle, EqPoint, MarketEvent, Trade } from "./Charts";
import { EVENT_LEGEND } from "./Charts";

const PriceChart = dynamic(() => import("./Charts").then((m) => m.PriceChart), { ssr: false });
const EquityChart = dynamic(() => import("./Charts").then((m) => m.EquityChart), { ssr: false });

type State = { job: any; initial: number; tf: number; candleTf: number; candles: Candle[]; equity: EqPoint[]; trades: Trade[]; events: MarketEvent[] };

const TFS = [[60, "H1"], [240, "H4"], [1440, "D1"]] as const;

export default function JobView({ id }: { id: number }) {
  const [tf, setTf] = useState<number | null>(null);
  const [st, setSt] = useState<State | null>(null);
  const [showTrades, setShowTrades] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const cur = useRef({ candleFrom: 0, eqFrom: 0, tradeFrom: 0, eventFrom: 0 });

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    cur.current = { candleFrom: 0, eqFrom: 0, tradeFrom: 0, eventFrom: 0 };
    setSt(null);
    const tick = async () => {
      const c = cur.current;
      const q = new URLSearchParams({ candleFrom: String(c.candleFrom), eqFrom: String(c.eqFrom), tradeFrom: String(c.tradeFrom) });
      if (tf) q.set("tf", String(tf));
      try {
        const r = await fetch(`/api/jobs/${id}?${q}`, { cache: "no-store" });
        const d = await r.json();
        if (stop || d.error) return;
        setSt((prev) => {
          const keep = <T,>(arr: T[] | undefined, from: number, key: (x: T) => number) => (arr ?? []).filter((x) => key(x) < from);
          const candles = [...keep(prev?.candles, c.candleFrom, (x) => x.time), ...d.candles];
          const equity = [...keep(prev?.equity, c.eqFrom, (x) => x.t), ...d.equity];
          const events = [...keep(prev?.events, c.candleFrom, (x) => x.t), ...d.events];
          const trades = [...(prev?.trades ?? []), ...d.trades];
          return { job: d.job, initial: d.initialBalance, tf: d.tf, candleTf: d.candleTf, candles, equity, trades, events };
        });
        // 次回は最後の足(未確定)から取り直す
        if (d.candles.length) c.candleFrom = d.candles[d.candles.length - 1].time;
        if (d.equity.length) c.eqFrom = d.equity[d.equity.length - 1].t;
        c.tradeFrom += d.trades.length;
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

  const tradeStats = useMemo(() => {
    const t = st?.trades ?? [];
    const wins = t.filter((x) => x.pnl > 0).length;
    const hunted = t.filter((x) => x.reason === "hunted").length;
    return { n: t.length, winRate: t.length ? wins / t.length : null, hunted };
  }, [st]);

  if (!st) return <div className="empty">読み込み中…</div>;
  const j = st.job;
  const elapsedDays = j.sim_t && j.t_start ? (j.sim_t - j.t_start) / 86400 : null;
  const m = j.metrics;
  const hasCandles = st.candles.length > 0;

  return (
    <>
      <div className="sub"><Link href={`/runs/${j.run_id}`}>← #{j.run_id} {j.run_name}</Link></div>
      <h1>
        {j.strategy_label} <span className="muted">on</span> {j.market_label} <span className="muted">seed {j.seed} / {j.broker}</span>{" "}
        <Status s={j.status} />
      </h1>
      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "8px 0 12px" }}>
        <div style={{ flex: 1 }}><Progress value={j.bars_total ? j.bars_done / j.bars_total : 0} /></div>
        <span className="num muted">{j.bars_done ?? 0}/{j.bars_total ?? 0} 本</span>
      </div>

      <div className="tiles">
        <div className="tile"><div className="k">シミュレーション日時</div><div className="v" style={{ fontSize: 15 }}>{date(j.sim_t)}</div></div>
        <div className="tile"><div className="k">経過</div><div className="v">{elapsedDays != null ? `${fmt(elapsedDays, 0)} 日` : "–"}</div></div>
        <div className="tile"><div className="k">取引回数</div><div className="v">{j.n_trades ?? 0}</div></div>
        <div className="tile"><div className="k">残高(円)</div><div className="v">{fmt(j.balance, 0)}</div></div>
        <div className="tile"><div className="k">有効証拠金(円)</div><div className="v">{fmt(j.equity, 0)}</div></div>
        <div className="tile"><div className="k">損益(円)</div><div className={`v ${tone(j.pnl_live)}`}>{yen(j.pnl_live)}</div></div>
        <div className="tile"><div className="k">勝率</div><div className="v">{pct(tradeStats.winRate)}</div></div>
        <div className="tile"><div className="k">ブローカーに狩られた</div><div className="v">{m?.hunted ?? tradeStats.hunted}</div></div>
      </div>

      {hasCandles ? (
        <>
          <h2>価格</h2>
          <div className="controls">
            {TFS.filter(([v]) => v >= st.candleTf).map(([v, l]) => (
              <button key={v} className={st.tf === v ? "on" : ""} onClick={() => setTf(v)}>{l}</button>
            ))}
            <span style={{ width: 12 }} />
            <button className={showTrades ? "on" : ""} onClick={() => setShowTrades(!showTrades)}>売買</button>
            <button className={showEvents ? "on" : ""} onClick={() => setShowEvents(!showEvents)}>悪意イベント</button>
          </div>
          <PriceChart candles={st.candles} trades={st.trades} events={st.events} tf={st.tf} showTrades={showTrades} showEvents={showEvents} />
          {showEvents && (
            <div className="legend">
              {Object.entries(EVENT_LEGEND).map(([k, v]) => (
                <span key={k}><i style={{ background: v.color, borderRadius: 2 }} />{v.text} = {v.label}</span>
              ))}
              <span>買/売 = 新規、利/損/狩 = 利確/損切り/ブローカーのストップ狩り</span>
            </div>
          )}
        </>
      ) : (
        <div className="panel muted" style={{ marginTop: 14 }}>
          このパターンはローソク足を保存していません(ローソク足と取引は、設定の detail_seeds のシードだけ保存します)。資産曲線のみ表示します。
        </div>
      )}

      <h2>資産曲線</h2>
      <EquityChart equity={st.equity} initial={st.initial} />

      {m && (
        <>
          <h2>最終成績</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr><th>損益(円)</th><th>CAGR</th><th>シャープ</th><th>素のシャープ</th><th>最大DD</th><th>PF</th><th>勝率</th><th>取引/年</th><th>素のpip/年</th><th>コストpip/年</th><th>スワップ(円)</th><th className="l">決済理由</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td className={`num ${tone(j.pnl)}`}>{yen(j.pnl)}</td>
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
        </>
      )}

      {st.trades.length > 0 && (
        <>
          <h2>直近の取引</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr><th>#</th><th className="l">売買</th><th className="l">建て</th><th className="l">決済</th><th>pip</th><th>コストpip</th><th>損益(円)</th><th className="l">理由</th><th>保有(本)</th></tr>
              </thead>
              <tbody>
                {st.trades.slice(-30).reverse().map((t: any) => (
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
      )}
    </>
  );
}
