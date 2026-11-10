"""コマンドライン。``python -m fxsim <command>``。

    list                         登録済みの市場モデル・戦略・ブローカー
    run CONFIG.yaml              実験を回して SQLite に保存
    eval-market MODEL            市場の案を評価(統計的性質 + 戦略パネル)
    ablate MODEL                 部品を 1 つずつ外して、どの悪意が効いているか見る
    eval-strategy NAME           戦略を「実データ + ランダム相場の複数パターン」で評価
    demo                         画面(web/)で追うための、ゆっくり進むシミュレーション
    report [RUN_ID]              実験結果を Markdown で表示(省略で最新)
    leaderboard                  全実験の横断ランキング
    check-lookahead [NAME...]    戦略が未来を見ていないか検査
    cases list                   動作ケース・テストケースと最新の結果
    cases run [ID...] [--all]    テストケースを機械的に実行して記録(定義とコードが同じなら飛ばす)
    cases show ID                テストケースの最新結果
    init-db                      空の DB を作る
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from . import catalog  # noqa: F401
from .backtest import BROKER_PRESETS
from .markets import MARKETS
from .paths import DEFAULT_DB
from .storage import connect, init_db, queries
from .strategies import STRATEGIES

PANEL = ["random_entry", "ma_cross", "donchian_breakout", "bollinger_reversion", "simple_follower"]
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
    from .reports import run_report

    run_id = run_experiment(ExperimentConfig.from_dict(cfg_dict), args.db)
    print()
    print(run_report(connect(args.db), run_id, patterns=getattr(args, "patterns", False)))
    return run_id


def _experiment_base(args, name: str, kind: str) -> dict:
    return {"name": args.name or name, "kind": kind, "proposal": args.proposal, "years": args.years,
            "seeds": args.seeds, "workers": args.workers, "brokers": [b for b in args.brokers.split(",") if b]}


# ---- コマンド -----------------------------------------------------------------

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

    con = connect(args.db)
    print(run_report(con, args.run_id or queries.latest_run_id(con), patterns=args.patterns))


def cmd_leaderboard(args) -> None:
    from .reports import leaderboard

    print(leaderboard(connect(args.db), args.kind, args.limit))


def cmd_check_lookahead(args) -> None:
    from .backtest.lookahead import check_lookahead

    md = MARKETS.create("malicious").generate(years=1, seed=11)
    bad = 0
    for name in args.names or STRATEGIES.names():
        tf = STRATEGIES.create(name).timeframe
        probs = check_lookahead(lambda name=name: STRATEGIES.create(name), md.bars(tf))
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


def cmd_init_db(args) -> None:
    init_db(args.db).close()
    print(f"initialized {args.db}")


# ---- 引数 -------------------------------------------------------------------

def _db(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=str(DEFAULT_DB))


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
    p.add_argument("run_id", type=int, nargs="?")
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

    p = add("init-db", cmd_init_db)
    _db(p)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
