"""コマンドライン。``python -m fxsim <command>``。

    list                         登録済みの市場モデル・戦略・ブローカー
    run CONFIG.yaml              実験を回して SQLite に保存
    eval-market MODEL            市場の案を評価(統計的性質 + 戦略パネル)
    ablate MODEL                 部品を 1 つずつ外して、どの悪意が効いているか見る
    eval-strategy NAME           戦略を「実データ + ランダム相場の複数パターン」で評価
    demo                         画面(web/)で追うための、ゆっくり進むシミュレーション
    report RUN_ID                実験結果を Markdown で表示
    leaderboard                  全実験の横断ランキング
    check-lookahead [NAME...]    戦略が未来を見ていないか検査
    cases list                   動作ケース・テストケースと最新の結果
    cases run [ID...] [--all]    テストケースを機械的に実行して記録(定義とコードが同じなら飛ばす)
    cases show ID                テストケースの最新結果
"""
from __future__ import annotations

import argparse
import json
import sys

from . import catalog  # noqa: F401
from .backtest import BROKER_PRESETS
from .markets import MARKETS
from .storage import DEFAULT_DB, connect, init_db
from .strategies import STRATEGIES

PANEL = ["random_entry", "ma_cross", "donchian_breakout", "bollinger_reversion", "simple_follower"]
REAL_SET = ["real", "real_bridge", "real_bootstrap", "real_bridge_malicious"]


def _seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def _common(p: argparse.ArgumentParser, years: float = 3, seeds: str = "1-3") -> None:
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--years", type=float, default=years)
    p.add_argument("--seeds", type=_seeds, default=_seeds(seeds), help="例: 1-5 / 1,3,7")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--proposal")
    p.add_argument("--brokers", default="fair,malicious")


def cmd_list(args) -> None:
    print("## 市場モデル")
    for e in MARKETS.entries():
        m = MARKETS.create(e.name)
        print(f"- {e.name} v{e.version} [{', '.join(e.tags)}]: {e.description}")
        print(f"    部品: {' -> '.join(c.key for c in m.components)}")
    print("\n## 戦略")
    for e in STRATEGIES.entries():
        print(f"- {e.name} v{e.version}: {e.description}")
    print("\n## ブローカー")
    for k, v in BROKER_PRESETS.items():
        print(f"- {k}: {v}")


def _run(cfg_dict: dict, args) -> int:
    from .experiments import ExperimentConfig, run_experiment
    from .experiments.report import run_report

    cfg = ExperimentConfig.from_dict(cfg_dict)
    run_id = run_experiment(cfg, args.db)
    con = connect(args.db)
    print()
    print(run_report(con, run_id, patterns=getattr(args, "patterns", False)))
    return run_id


def _base(args, name: str, kind: str) -> dict:
    return {"name": args.name or name, "kind": kind, "proposal": args.proposal, "years": args.years,
            "seeds": args.seeds, "workers": args.workers,
            "brokers": [b for b in args.brokers.split(",") if b]}


def cmd_run(args) -> None:
    import yaml

    with open(args.config, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for k in ("years", "workers"):
        v = getattr(args, k)
        if v:
            d[k] = v
    if args.seeds:
        d["seeds"] = args.seeds
    _run(d, args)


def cmd_eval_market(args) -> None:
    params = json.loads(args.params) if args.params else {}
    d = _base(args, f"eval-market {args.model}", "eval_market")
    d["markets"] = [{"model": "random_walk", "label": "random_walk"},
                    {"model": args.model, "label": args.label or args.model, "params": params}]
    d["strategies"] = PANEL
    _run(d, args)


def cmd_ablate(args) -> None:
    params = json.loads(args.params) if args.params else {}
    model = MARKETS.create(args.model, params)
    d = _base(args, f"ablate {args.model}", "ablation")
    markets = [{"model": args.model, "label": "full", "params": params}]
    for c in model.components:
        if c.key in ("spread", "source"):
            continue
        markets.append({"model": args.model, "label": f"-{c.key}", "params": {**params, c.key: None}})
    d["markets"] = markets
    d["strategies"] = PANEL
    _run(d, args)


def cmd_eval_strategy(args) -> None:
    params = json.loads(args.params) if args.params else {}
    d = _base(args, f"eval-strategy {args.name_}", "eval_strategy")
    d["markets"] = [m for m in (args.markets.split(",") if args.markets else REAL_SET)]
    d["strategies"] = [{"name": args.name_, "params": params}] + (["random_entry"] if args.null else [])
    d["detail_seeds"] = args.seeds[:1]
    _run(d, args)


def cmd_demo(args) -> None:
    d = {"name": args.name or "live demo", "kind": "demo", "years": args.years, "seeds": [args.seed],
         "workers": 1, "brokers": ["malicious"],
         "markets": [{"model": args.model}],
         "strategies": [{"name": args.strategy}],
         "storage": {"candle_timeframe": 60},
         "live": {"flush_every": args.flush_every, "delay_ms": args.delay_ms}}
    _run(d, args)


def cmd_report(args) -> None:
    from .experiments.report import run_report

    con = connect(args.db)
    run_id = args.run_id
    if run_id is None:
        row = con.execute("SELECT MAX(id) FROM runs").fetchone()
        run_id = row[0]
    print(run_report(con, run_id, patterns=args.patterns))


def cmd_leaderboard(args) -> None:
    from .experiments.report import leaderboard

    print(leaderboard(connect(args.db), args.kind, args.limit))


def cmd_check_lookahead(args) -> None:
    from .backtest.lookahead import check_lookahead

    md = MARKETS.create("malicious").generate(years=1, seed=11)
    bad = 0
    for name in args.names or STRATEGIES.names():
        tf = STRATEGIES.create(name).timeframe
        probs = check_lookahead(lambda: STRATEGIES.create(name), md.bars(tf))
        print(f"{name}: {'OK' if not probs else 'NG ' + '; '.join(probs)}")
        bad += bool(probs)
    sys.exit(1 if bad else 0)


def cmd_cases(args) -> None:
    from .cases import list_cases, run_cases
    from .experiments.report import run_report

    con = init_db(args.db)
    if args.action == "list":
        print(list_cases(con))
    elif args.action == "run":
        if not args.ids and not args.all:
            sys.exit("テストケースの ID を指定するか --all を付けてください")
        run_cases(con, args.db, args.ids or None, force=args.force)
        print()
        print(list_cases(con))
    elif args.action == "show":
        if not args.ids:
            sys.exit("テストケースの ID を指定してください")
        row = con.execute("SELECT run_id FROM case_runs WHERE test_id=? AND status='done' ORDER BY id DESC LIMIT 1",
                          (args.ids[0],)).fetchone()
        if not row:
            sys.exit(f"{args.ids[0]} はまだ実行されていません")
        print(run_report(con, row[0], patterns=True))


def cmd_init_db(args) -> None:
    init_db(args.db).close()
    print(f"initialized {args.db}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fxsim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--years", type=float)
    p.add_argument("--seeds", type=_seeds)
    p.add_argument("--workers", type=int)
    p.add_argument("--patterns", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eval-market")
    p.add_argument("model")
    p.add_argument("--params", help="JSON。例: '{\"stop_hunt\": {\"prob\": 0.1}}'")
    p.add_argument("--label")
    _common(p)
    p.set_defaults(func=cmd_eval_market)

    p = sub.add_parser("ablate")
    p.add_argument("model")
    p.add_argument("--params")
    _common(p)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("eval-strategy")
    p.add_argument("name_", metavar="strategy")
    p.add_argument("--params")
    p.add_argument("--markets", help=f"カンマ区切り。既定: {','.join(REAL_SET)}")
    p.add_argument("--null", action="store_true", help="比較用に random_entry も回す")
    p.add_argument("--patterns", action="store_true", help="パターンごとの損益も表示")
    _common(p, years=5, seeds="1-10")
    p.set_defaults(func=cmd_eval_strategy)

    p = sub.add_parser("demo")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--model", default="malicious")
    p.add_argument("--strategy", default="donchian_breakout")
    p.add_argument("--years", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delay-ms", type=float, default=200)
    p.add_argument("--flush-every", type=int, default=24)
    p.add_argument("--name")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("report")
    p.add_argument("run_id", type=int, nargs="?")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--patterns", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("leaderboard")
    p.add_argument("--kind", choices=["strategies", "markets"], default="strategies")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("check-lookahead")
    p.add_argument("names", nargs="*")
    p.set_defaults(func=cmd_check_lookahead)

    p = sub.add_parser("cases")
    p.add_argument("action", choices=["list", "run", "show"])
    p.add_argument("ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("init-db")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.set_defaults(func=cmd_init_db)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
