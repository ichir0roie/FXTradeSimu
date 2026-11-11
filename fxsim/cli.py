"""コマンドライン。``python -m fxsim <command>``。

    list                         登録済みの市場モデル・戦略・ブローカー
    run CONFIG.yaml              実験を回して SQLite に保存
    eval-market MODEL            市場の案を評価(統計的性質 + 戦略パネル)
    ablate MODEL                 部品を 1 つずつ外して、どの悪意が効いているか見る
    eval-strategy NAME           戦略を「実データ + ランダム相場の複数パターン」で評価
    demo                         画面(web/)で追うための、ゆっくり進むシミュレーション
    report [RUN_ID|RUN_KEY]      実験結果を Markdown で表示(省略で最新)
    leaderboard                  全実験の横断ランキング
    check-lookahead [NAME...]    戦略が未来を見ていないか検査
    cases list                   動作ケース・テストケースと最新の結果
    cases run [ID...] [--all]    テストケースを機械的に実行して記録(定義とコードが同じなら飛ばす)
    cases show ID                テストケースの最新結果
    verify BEHAVIOR... | --all   動作ケースを標準パターンで長期検証し、合否を判定して記録
    pair-eval BEHAVIOR... | --all  実データだけ(主要ペア x 期間)で、方向でたらめ版と対にして評価する(E016)
    summary [RUN...] [--all]     実験の概要を docs/results/<run_key>/ に書き出す(既定の DB への実験は終わったら自動)
    init-db                      空の DB を作る
    compact RUN_ID...            保存済みの資産曲線を粗い間隔に間引き、DB を詰める
    fetch-fred                   FRED から対円 7 ペアの日足と 8 通貨の短期金利を取って data/real/ に書く
    fetch-mt5 CONFIG             MT5 の足を相場の DB(data/market.sqlite)に書き足す(configs/market_data/*.yaml)
    market-data                  相場の DB に溜まっている足の一覧(読むときは実データのパスに db:USDJPY/M1)
    overfit RUN                  実験の中の戦略の集まりの PBO と Deflated Sharpe(試した数の補正)
    trade run CONFIG             MT5 の口座で動作ケースの戦略を常駐して動かす(configs/trading/*.yaml、docs/TRADING.md)
    trade check CONFIG           MT5 につなぎ、口座・銘柄・建玉と、いまの足での判断を見る(注文は出さない)
    trade status CONFIG          常駐の状態と直近の記録(data/trading/<name>/)
    trade-replay CONFIG...       常駐を相場の DB の 1 分足の再生の上で 24 時間動かし、動作を確かめる(故障も入れる)
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from . import catalog  # noqa: F401
from .backtest import BROKER_PRESETS
from .markets import MARKETS
from .paths import DEFAULT_DB, MARKET_DB, SUMMARIES_DIR
from .storage import connect, init_db, queries, vacuum
from .strategies import STRATEGIES

# 悪用されにくさを測るパネル。rsi_reversion(1 時間足のオシレーター、R008)と scalping(5 分足、M006)は、
# 1 時間足の 5 本では見えなかった穴を映すために足した
PANEL = ["random_entry", "ma_cross", "donchian_breakout", "bollinger_reversion", "simple_follower", "rsi_reversion",
         "scalping"]
REAL_SET = ["real", "real_bridge", "real_bootstrap", "real_bridge_malicious"]


def parse_seeds(s: str) -> list[int]:
    """'1-5' / '1,3,7' / '4'。"""
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def _json(s: str | None) -> dict:
    return json.loads(s) if s else {}


# ---- 実験を回す共通処理 --------------------------------------------------------

def _run_and_report(cfg_dict: dict, args) -> int:
    from .experiments import ExperimentConfig, run_experiment
    from .reports import auto_summary, run_report

    run_id = run_experiment(ExperimentConfig.from_dict(cfg_dict), args.db)
    con = connect(args.db)
    auto_summary(con, args.db, run_id)
    print()
    print(run_report(con, run_id, patterns=getattr(args, "patterns", False)))
    return run_id


# コマンドから組み立てる実験(eval-market / ablate / eval-strategy)の保存。目的は性質と成績の統計なので粗くする
# (既定のままだと 1 時間足の資産曲線と取引の明細で、10 年 x 8 の eval-market 1 本が 70MB を超えた)
CLI_STORAGE = {"candle_timeframe": 1440, "equity_detail_timeframe": 1440, "equity_timeframe": 10080,
               "trades": "none", "events": False}


def _experiment_base(args, name: str, kind: str) -> dict:
    return {"name": args.name or name, "kind": kind, "proposal": args.proposal, "years": args.years,
            "seeds": args.seeds, "workers": args.workers, "brokers": [b for b in args.brokers.split(",") if b],
            "storage": dict(CLI_STORAGE)}


# ---- コマンド -----------------------------------------------------------------

def cmd_list(args) -> None:
    print("## 市場モデル")
    for e in MARKETS.entries():
        m = MARKETS.create(e.name)
        print(f"- {e.name} v{e.version} [{', '.join(e.tags)}]: {e.description}")
        print(f"    部品: {' -> '.join(c.key for c in getattr(m, 'components', []))}")
    for tier, title in (("main", "本命の戦略(MT5 のテスターで回す)"), ("basic", "基本検証の戦略")):
        print(f"\n## {title}")
        for e in STRATEGIES.entries():
            if getattr(e.factory, "tier", "basic") == tier:
                print(f"- {e.name} v{e.version}: {e.description}")
    print("\n## ブローカー")
    for k, v in BROKER_PRESETS.items():
        print(f"- {k}: {v}")


def cmd_run(args) -> None:
    with open(args.config, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for k in ("years", "workers", "seeds"):
        if getattr(args, k):
            d[k] = getattr(args, k)
    _run_and_report(d, args)


def cmd_eval_market(args) -> None:
    d = _experiment_base(args, f"eval-market {args.model}", "eval_market")
    target = {"model": args.model, "label": args.label or args.model, "params": _json(args.params)}
    d["markets"] = ["random_walk", target]
    d["strategies"] = PANEL
    _run_and_report(d, args)


def cmd_ablate(args) -> None:
    params = _json(args.params)
    d = _experiment_base(args, f"ablate {args.model}", "ablation")
    d["markets"] = [{"model": args.model, "label": "full", "params": params}] + [
        {"model": args.model, "label": f"-{c.key}", "params": {**params, c.key: None}}
        for c in MARKETS.create(args.model, params).components if c.key not in ("spread", "source")]
    d["strategies"] = PANEL
    _run_and_report(d, args)


def cmd_eval_strategy(args) -> None:
    d = _experiment_base(args, f"eval-strategy {args.strategy}", "eval_strategy")
    d["markets"] = args.markets.split(",") if args.markets else REAL_SET
    d["strategies"] = [{"name": args.strategy, "params": _json(args.params)}] + (["random_entry"] if args.null else [])
    d["detail_seeds"] = args.seeds[:1]
    _run_and_report(d, args)


def cmd_demo(args) -> None:
    _run_and_report({
        "name": args.name or "live demo", "kind": "demo", "years": args.years, "seeds": [args.seed], "workers": 1,
        "brokers": ["malicious"], "markets": [args.model], "strategies": [args.strategy],
        "storage": {"candle_timeframe": 60},
        "live": {"flush_every": args.flush_every, "delay_ms": args.delay_ms},
    }, args)


def cmd_report(args) -> None:
    from .reports import run_report

    con = init_db(args.db)
    print(run_report(con, queries.resolve_run(con, args.run_id) or queries.latest_run_id(con),
                     patterns=args.patterns))


def cmd_leaderboard(args) -> None:
    from .reports import leaderboard

    print(leaderboard(init_db(args.db), args.kind, args.limit))


def cmd_check_lookahead(args) -> None:
    from .backtest.lookahead import check_lookahead
    md = MARKETS.create("malicious").generate(years=1, seed=11)
    basket = None
    bad = 0
    for name in args.names or STRATEGIES.names():
        strat = STRATEGIES.create(name)
        tf = strat.timeframe
        if strat.multi:
            if basket is None:     # 複数ペアの戦略は、見る期間が長いので 10 年の並べ替えた相場で
                basket = MARKETS.create("basket_bootstrap").generate(years=10, seed=11).meta["legs"]
            bars = {p: leg.bars(tf) for p, leg in basket.items()}
        else:
            bars = md.bars(tf)
        probs = check_lookahead(lambda name=name: STRATEGIES.create(name), bars)
        print(f"{name}: {'OK' if not probs else 'NG ' + '; '.join(probs)}")
        bad += bool(probs)
    sys.exit(1 if bad else 0)


def cmd_cases(args) -> None:
    from .cases import run_cases
    from .reports import cases_overview, run_report

    con = init_db(args.db)
    if args.action == "list":
        print(cases_overview(con))
    elif args.action == "run":
        if not args.ids and not args.all:
            sys.exit("テストケースの ID を指定するか --all を付けてください")
        run_cases(con, args.db, args.ids or None, force=args.force)
        print()
        print(cases_overview(con))
    else:   # show
        if not args.ids:
            sys.exit("テストケースの ID を指定してください")
        latest = queries.latest_case_run(con, args.ids[0])
        if not latest:
            sys.exit(f"{args.ids[0]} はまだ実行されていません")
        print(run_report(con, latest[0], patterns=True))
        if latest[1].get("verdict"):
            from .verification import format_verdict

            print()
            print(format_verdict(latest[1]["verdict"]))


def cmd_verify(args) -> None:
    from .cases import run_cases, verification_case
    from .cases.loader import load_cases
    from .reports import run_report
    from .verification import format_verdict

    con = init_db(args.db)
    book = load_cases()
    ids = sorted(book.behaviors) if args.all else args.behaviors
    if not ids:
        sys.exit("動作ケースの ID を指定するか --all を付けてください")
    verdicts = []
    for bid in ids:
        if bid not in book.behaviors:
            sys.exit(f"unknown behavior {bid}. known: {sorted(book.behaviors)}")
        if args.rejudge:
            v = _rejudge(con, args.db, bid)
            if v:
                verdicts.append((bid, book.behaviors[bid].strategy, v))
            continue
        v = book.behaviors[bid].verify_settings(years=args.years, seeds=args.seeds, block_days=args.block_days)
        t = verification_case(bid, years=v["years"], seeds=v["seeds"], start=args.start, end=args.end,
                              block_days=v["block_days"], dataset=args.dataset)
        if args.workers:
            t.workers = args.workers
        run_cases(con, args.db, [t.id], force=args.force, extra=[t])
        latest = queries.latest_case_run(con, t.id)
        if not latest:
            continue
        run_id, summary = latest
        verdict = summary.get("verdict")
        if len(ids) == 1:
            print()
            print(run_report(con, run_id, patterns=args.patterns))
        if verdict:
            print()
            print(format_verdict(verdict))
            verdicts.append((bid, book.behaviors[bid].strategy, verdict))
    if len(verdicts) > 1:
        keys = [c["key"] for c in verdicts[0][2]["checks"]]
        print("\n## まとめ\n\n| 動作ケース | 戦略 | 判定 | " + " | ".join(keys) + " |\n|---|---|---|"
              + "---|" * len(keys))
        for bid, strat, v in verdicts:
            ok = {c["key"]: "✓" if c["ok"] else "✗" for c in v["checks"]}
            print(f"| {bid} | {strat} | {v['verdict']} | " + " | ".join(ok.get(k, "-") for k in keys) + " |")


def cmd_pair_eval(args) -> None:
    """実データ(ペア x 期間)だけでの評価(fxsim/pair_eval.py、E016)。"""
    from .cases import run_cases
    from .cases.loader import load_cases
    from .cases.runner import summarize_run
    from .pair_eval import compute_pair_verdict, format_pair_verdict, pair_eval_case
    from .storage import writer

    con = init_db(args.db)
    book = load_cases()
    ids = sorted(book.behaviors) if args.all else args.behaviors
    if not ids:
        sys.exit("動作ケースの ID を指定するか --all を付けてください")
    results = []
    for bid in ids:
        if bid not in book.behaviors:
            sys.exit(f"unknown behavior {bid}. known: {sorted(book.behaviors)}")
        b = book.behaviors[bid]
        t = pair_eval_case(b, pairs=args.pairs, broker=args.broker, daily_fill_hour=args.daily_fill_hour,
                           swap_side=args.swap_side)
        if args.workers:
            t.workers = args.workers
        if not args.rejudge:
            run_cases(con, args.db, [t.id], force=args.force, extra=[t])
        row = queries.latest_case_run_row(con, t.id)
        if row is None or row["run_id"] is None:
            print(f"{t.id}: 結果がありません")
            continue
        v = compute_pair_verdict(con, row["run_id"], bid)
        summary = summarize_run(con, row["run_id"])
        summary["pair_verdict"] = v
        writer.update_case_summary(con, row["id"], summary)
        print()
        print(format_pair_verdict(v))
        results.append((b, v))
    if len(results) > 1:
        keys = [c["key"] for c in results[0][1]["checks"]]
        print("\n## まとめ\n\n| 動作ケース | 戦略 | 足 | 判定 | " + " | ".join(keys)
              + " | 損益の中央値 | t | 帰無との差の t | 取引 / 年 | 素の損益 / 取引 | 費用 / 取引 |\n|---|---|---:|---|"
              + "---|" * len(keys) + "---:|---:|---:|---:|---:|---:|")
        for b, v in results:
            ok = {c["key"]: "o" if c["ok"] else "x" for c in v["checks"]}
            pnl, paired = v["pnl"], v["paired"]
            print(f"| {b.id} | {b.strategy} | {b.strategy_params().get('timeframe')} | {v['verdict']} | "
                  + " | ".join(ok[k] for k in keys)
                  + f" | {pnl.get('median', 0):+,.0f} | {pnl.get('t', 0):+.2f} | {paired.get('t', 0):+.2f}"
                  f" | {v['trades_per_year']:.0f} | {v['gross_pips_per_trade']:+.2f}"
                  f" | {v['cost_pips_per_trade']:.2f} |")


def _rejudge(con, db, bid: str) -> dict | None:
    """記録済みの最新の検証の判定を、今の基準で付け直す(売買は回し直さない)。概要(docs/results)も書き直す。"""
    from .cases.runner import summarize_run
    from .reports.summary import auto_summary
    from .storage import writer
    from .verification import format_verdict

    row = queries.latest_case_run_row(con, f"V-{bid}")
    if row is None:
        print(f"V-{bid}: まだ検証していません")
        return None
    summary = summarize_run(con, row["run_id"], verify_label=bid)
    writer.update_case_summary(con, row["id"], summary)
    auto_summary(con, db, row["run_id"], log=lambda *a: None)
    print()
    print(format_verdict(summary["verdict"]))
    return summary["verdict"]


def cmd_fetch_fred(args) -> None:
    from .data.fred import fetch

    fetch()


def cmd_fetch_mt5(args) -> None:
    import logging

    from .trading.history import load_fetch_config, run_fetch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_fetch_config(args.config)
    if args.symbols:
        cfg.symbols = args.symbols
    if args.timeframes:
        cfg.timeframes = args.timeframes
    if args.bars:
        cfg.bars = args.bars
    cfg.__post_init__()
    for rec in run_fetch(cfg, args.market_db):
        link = {None: "first fetch", 1: "connected", 0: "GAP before this fetch"}[rec["connected"]]
        print(f"{rec['source']} {rec['symbol']} M{rec['timeframe']}: {rec['n_rows']} bars "
              f"({_utc(rec['first_time'])} .. {_utc(rec['last_time'])} UTC), new {rec['n_new']}, {link}")


def cmd_market_data(args) -> None:
    from .storage import prices

    con = prices.open_market_db(args.market_db)
    rows = prices.coverage(con)
    if not rows:
        print(f"no rates in {args.market_db} (python -m fxsim fetch-mt5 CONFIG)")
        return
    print("| source | symbol | tf | bars | first (UTC) | last (UTC) | last fetched |")
    print("|---|---|---|---:|---|---|---|")
    for r in rows:
        print(f"| {r['source']} | {r['symbol']} | {prices.TIMEFRAMES[r['timeframe']].upper()} | {r['n']} | "
              f"{_utc(r['first_time'])} | {_utc(r['last_time'])} | {r['last_fetched_at']} |")


def _utc(sec: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(sec), timezone.utc).strftime("%Y-%m-%d %H:%M")


def cmd_overfit(args) -> None:
    from .overfit import format_overfit, run_overfit

    con = init_db(args.db)
    run_id = queries.resolve_run(con, args.run_id) or queries.latest_run_id(con)
    print(format_overfit(run_overfit(con, run_id, market=args.market, broker=args.broker, n_blocks=args.blocks)))


def cmd_summary(args) -> None:
    """実験の概要(docs/results/<run_key>/)を書き出す。既定の DB への実験は終わったら自動で書き出される。"""
    from .reports import export_summary

    con = init_db(args.db)
    if args.all:
        ids = [r[0] for r in con.execute("SELECT id FROM runs WHERE kind != 'demo' AND status != 'running'")]
    else:
        ids = [queries.resolve_run(con, r) for r in args.runs] or [queries.latest_run_id(con)]
    for run_id in ids:
        print(f"run #{run_id} -> {export_summary(con, run_id, args.out)}")


def cmd_init_db(args) -> None:
    init_db(args.db).close()
    print(f"initialized {args.db}")


def cmd_compact(args) -> None:
    import os

    from .storage import writer

    con = init_db(args.db)
    before = os.path.getsize(args.db)
    run_ids = [queries.resolve_run(con, r) for r in args.run_ids]
    for run_id in run_ids:
        n = writer.compact_equity(con, run_id, args.equity_minutes, args.detail_equity_minutes)
        print(f"run #{run_id}: removed {n} equity points")
    vacuum(con)
    con.close()
    print(f"{args.db}: {before / 1e6:.1f} MB -> {os.path.getsize(args.db) / 1e6:.1f} MB")


def cmd_trade(args) -> None:
    import logging
    from dataclasses import asdict

    from .trading import load_trader_config
    from .trading.service import Service

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_trader_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    if args.action == "status":
        from .trading.trader import TraderState
        for d in (cfg.state_dir, cfg.state_dir.with_name(cfg.state_dir.name + ".dry")):
            if not (d / "state.json").exists():
                continue
            print(f"## {d}")
            print(json.dumps(asdict(TraderState.load(d / "state.json")), ensure_ascii=False, indent=2))
            journal = d / "journal.jsonl"
            lines = journal.read_text(encoding="utf-8").splitlines() if journal.exists() else []
            for line in lines[-args.tail:]:
                rec = json.loads(line)
                if rec["kind"] == "bar" and "positions" in rec["sim"]:        # 複数ペア
                    held = {k: v["side"] for k, v in rec["sim"]["positions"].items() if v}
                    acts = ", ".join(f"{a['symbol']} {a['action']['kind']} {a['action']['side']:+d} "
                                     f"{a['action']['lots']}" for a in rec["actions"]) or "-"
                    print(f"{rec['bar_close_utc']}  sim {held or 'flat'}  actions: {acts}")
                elif rec["kind"] == "bar":
                    acts = ", ".join(f"{a['kind']} {a['side']:+d} {a['lots']}" for a in rec["actions"]) or "-"
                    pos = rec["sim"]["position"]
                    sim = f"{pos['side']:+d} {pos['units']:.0f}" if pos else "flat"
                    print(f"{rec['bar_close_utc']}  sim {sim:>12}  real {len(rec['real'])}  actions: {acts}")
                else:
                    rest = {k: v for k, v in rec.items() if k != "kind"}
                    print(f"[{rec['kind']}] {json.dumps(rest, ensure_ascii=False, default=str)[:200]}")
        return
    svc = Service(cfg)
    if args.action == "check":
        cfg.dry_run = True
        trader = svc.connect()
        c = svc.client
        print("server offset (h):", (c.server_offset(cfg.symbol) or 0) / 3600 if cfg.server_utc_offset_hours == "auto"
              else cfg.server_utc_offset_hours)
        forming = int(c.rates(cfg.symbol, trader.timeframe, 1)["time"][-1])
        if cfg.symbols:                          # 複数ペア
            decided = trader.decide(forming)
            if decided is None:
                print("some symbols have no forming bar yet; try again later")
            else:
                snap, per, meta = decided
                print(json.dumps({"meta": meta, "sim_equity": snap.equity, "server_hour": trader.server_hour(),
                                  "fill_server_hour": cfg.fill_server_hour,
                                  "pairs": {s: {"sim_position": asdict(snap.pairs[s[:6].lower()].position)
                                                if snap.pairs[s[:6].lower()].position else None,
                                                "real": [asdict(p) for p in per[s][0]],
                                                "actions": [asdict(a) for a in per[s][1]]} for s in cfg.symbols}},
                                 ensure_ascii=False, indent=2, default=str))
            c.shutdown()
            return
        snap, real, actions, meta = trader.decide(forming)
        print(json.dumps({"meta": meta, "sim_position": asdict(snap.position) if snap.position else None,
                          "sim_orders": [asdict(o) for o in snap.orders], "real": [asdict(p) for p in real],
                          "actions": [asdict(a) for a in actions]}, ensure_ascii=False, indent=2, default=str))
        c.shutdown()
        return
    svc.install_signals()
    svc.run(once=args.once)


def cmd_trade_replay(args) -> None:
    from pathlib import Path

    from .trading.replay import ReplayOptions, run_replay

    manual = [json.loads(m) for m in args.manual]
    opts = ReplayOptions(start=args.start, end=args.end, faults=args.faults, seed=args.seed,
                         separate_accounts=args.separate_accounts, balance=args.balance, source=args.source,
                         out_dir=Path(args.out) if args.out else None, manual=manual)
    r = run_replay(args.configs, opts, market_db=args.market_db)
    print((Path(r["out_dir"]) / "report.md").read_text(encoding="utf-8"))


# ---- 引数 -------------------------------------------------------------------

def _db(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=str(DEFAULT_DB))


def _market_db(p: argparse.ArgumentParser) -> None:
    p.add_argument("--market-db", default=str(MARKET_DB), help="相場の DB(既定: data/market.sqlite)")


def _experiment_options(p: argparse.ArgumentParser, years: float = 3, seeds: str = "1-3") -> None:
    _db(p)
    p.add_argument("--years", type=float, default=years)
    p.add_argument("--seeds", type=parse_seeds, default=parse_seeds(seeds), help="例: 1-5 / 1,3,7")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--proposal")
    p.add_argument("--brokers", default="fair,malicious")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fxsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, func, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(func=func)
        return p

    add("list", cmd_list)

    p = add("run", cmd_run)
    p.add_argument("config")
    _db(p)
    p.add_argument("--years", type=float)
    p.add_argument("--seeds", type=parse_seeds)
    p.add_argument("--workers", type=int)
    p.add_argument("--patterns", action="store_true")

    p = add("eval-market", cmd_eval_market)
    p.add_argument("model")
    p.add_argument("--params", help="JSON。例: '{\"stop_hunt\": {\"prob\": 0.1}}'")
    p.add_argument("--label")
    _experiment_options(p)

    p = add("ablate", cmd_ablate)
    p.add_argument("model")
    p.add_argument("--params")
    _experiment_options(p)

    p = add("eval-strategy", cmd_eval_strategy)
    p.add_argument("strategy")
    p.add_argument("--params")
    p.add_argument("--markets", help=f"カンマ区切り。既定: {','.join(REAL_SET)}")
    p.add_argument("--null", action="store_true", help="比較用に random_entry も回す")
    p.add_argument("--patterns", action="store_true", help="パターンごとの損益も表示")
    _experiment_options(p, years=5, seeds="1-10")

    p = add("demo", cmd_demo)
    _db(p)
    p.add_argument("--model", default="malicious")
    p.add_argument("--strategy", default="donchian_breakout")
    p.add_argument("--years", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delay-ms", type=float, default=200)
    p.add_argument("--flush-every", type=int, default=24)
    p.add_argument("--name")

    p = add("report", cmd_report)
    p.add_argument("run_id", nargs="?", help="番号か run_key(前方一致も可)")
    _db(p)
    p.add_argument("--patterns", action="store_true")

    p = add("leaderboard", cmd_leaderboard)
    p.add_argument("--kind", choices=["strategies", "markets"], default="strategies")
    p.add_argument("--limit", type=int, default=30)
    _db(p)

    p = add("check-lookahead", cmd_check_lookahead)
    p.add_argument("names", nargs="*")

    p = add("cases", cmd_cases)
    p.add_argument("action", choices=["list", "run", "show"])
    p.add_argument("ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    _db(p)

    p = add("pair-eval", cmd_pair_eval)
    p.add_argument("behaviors", nargs="*", help="動作ケースの ID(cases/behaviors)。複数可")
    p.add_argument("--all", action="store_true", help="全部の動作ケース(定義とコードが前回と同じなら飛ばす)")
    p.add_argument("--pairs", nargs="+", help="ペアを絞る(既定は主要 10 ペア)")
    p.add_argument("--broker", help="動作ケースのブローカーを差し替える(例: oanda)。テストケースの ID に付く")
    p.add_argument("--swap-side", action="store_true",
                   help="スワップを受け取る向きの新規だけを出す(EngineConfig.swap_side_only、方向でたらめ版にも掛かる)")
    p.add_argument("--daily-fill-hour", type=int,
                   help="日足の約定のスプレッドに使うサーバーの時(既定 0 = ロールオーバー)。2 で 2 時間遅らせた近似")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rejudge", action="store_true", help="回し直さずに、記録済みの結果で判定だけやり直す")
    _db(p)

    p = add("verify", cmd_verify)
    p.add_argument("behaviors", nargs="*", help="動作ケースの ID(cases/behaviors)。複数可")
    p.add_argument("--all", action="store_true", help="全部の動作ケース(定義とコードが前回と同じなら飛ばす)")
    p.add_argument("--years", type=float, help="省略で動作ケースの verify、それも無ければ 30")
    p.add_argument("--seeds", type=parse_seeds, help="省略で動作ケースの verify、それも無ければ 1-10")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--block-days", type=float,
                   help="並べ替えのブロックの長さ(日)。戦略が見る期間より長くする。省略で動作ケースの verify か 250")
    p.add_argument("--dataset", help="実データ系に使う CSV(既定: data/real/usdjpy_d1_fred.csv)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--patterns", action="store_true")
    p.add_argument("--rejudge", action="store_true",
                   help="回し直さず、記録済みの最新の結果に今の判定の基準を当て直す(概要も書き直す)")
    _db(p)

    p = add("summary", cmd_summary, help="実験の概要を docs/results/<run_key>/ に書き出す")
    p.add_argument("runs", nargs="*", help="番号か run_key(省略で最新)")
    p.add_argument("--all", action="store_true", help="demo 以外の全実行")
    p.add_argument("--out", default=str(SUMMARIES_DIR))
    _db(p)

    p = add("overfit", cmd_overfit, help="戦略の集まりの PBO と Deflated Sharpe")
    p.add_argument("run_id", nargs="?", help="番号か run_key(省略で最新)")
    p.add_argument("--market", help="市場のラベル(省略で最初のもの)")
    p.add_argument("--broker", help="ブローカーのラベル(省略で最初のもの)")
    p.add_argument("--blocks", type=int, default=10, help="CSCV の塊の数(偶数)")
    _db(p)

    add("fetch-fred", cmd_fetch_fred, help="複数通貨ペアの実データと金利を FRED から取る")

    p = add("fetch-mt5", cmd_fetch_mt5, help="MT5 の足を相場の DB に書き足す")
    p.add_argument("config", help="configs/market_data/*.yaml")
    p.add_argument("--symbols", nargs="+", help="設定の symbols を差し替える")
    p.add_argument("--timeframes", nargs="+", help="設定の timeframes を差し替える(M1 M5 H1 ...)")
    p.add_argument("--bars", type=int, help="設定の bars を差し替える(99999 まで)")
    _market_db(p)

    p = add("market-data", cmd_market_data, help="相場の DB に溜まっている足の一覧")
    _market_db(p)

    p = add("init-db", cmd_init_db)
    _db(p)

    p = add("compact", cmd_compact)
    p.add_argument("run_ids", nargs="+", help="番号か run_key")
    p.add_argument("--equity-minutes", type=int, default=129600, help="詳細保存でないジョブの資産曲線の間隔(分)")
    p.add_argument("--detail-equity-minutes", type=int, default=None,
                   help="詳細保存のジョブの資産曲線の間隔(分。省略で間引かない)")
    _db(p)

    p = add("trade", cmd_trade, help="MT5 の口座で動作ケースの戦略を常駐して動かす")
    p.add_argument("action", choices=["run", "check", "status"])
    p.add_argument("config", help="configs/trading/*.yaml")
    p.add_argument("--dry-run", action="store_true", help="注文を出さず、出すはずの注文を記録するだけ")
    p.add_argument("--once", action="store_true", help="新しい足を 1 本処理したら終わる")
    p.add_argument("--tail", type=int, default=20, help="status: 表示する記録の行数")
    p.add_argument("--log-level", default="info")

    p = add("trade-replay", cmd_trade_replay, help="常駐を 1 分足の再生の上で動かして確かめる(docs/TRADING.md)")
    p.add_argument("configs", nargs="+", help="configs/trading/*.yaml(既定は同じ口座で一緒に動かす)")
    p.add_argument("--start", help="UTC(例 2026-08-17)。省略で 1 分足から作れる最初の足の後の月曜")
    p.add_argument("--end", help="UTC。省略で 1 分足の最後")
    p.add_argument("--faults", choices=["default", "none"], default="default")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--separate-accounts", action="store_true", help="常駐ごとに別の口座にする")
    p.add_argument("--balance", type=float, default=3_000_000.0, help="口座の初めの残高(円)")
    p.add_argument("--source", help="相場の DB の出どころ(省略で最初のもの)")
    p.add_argument("--out", help="記録の置き場所(既定 data/trading/replay/<時刻>/)")
    p.add_argument("--manual", action="append", default=[],
                   help='手動の建玉 JSON(例 \'{"at": "2026-09-01T03:00", "side": 1, "volume": 0.1, "sl_dist": 1.0}\')')
    _market_db(p)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
