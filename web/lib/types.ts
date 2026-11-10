// API(app/api/*)が返す形。サーバーとコンポーネントで共有する。

export type Status = "queued" | "running" | "done" | "failed" | "skipped";

export type Schedule = {
  decision_every: number;
  hours: [string, string][] | null;
  tz: string;
  days: string[];
  outside: "hold" | "close";
};

export type RunListItem = {
  id: number;
  name: string;
  kind: string;
  proposal: string | null;
  status: Status;
  created_at: string;
  years: number;
  git_sha: string | null;
  elapsed_sec: number | null;
  n_jobs: number;
  n_finished: number;
  n_paths: number;
  bar_progress: number | null;
};

export type StrategySummaryRow = {
  run_id: number;
  market_label: string;
  broker: string;
  strategy_label: string;
  n_seeds: number;
  sharpe: number | null;
  gross_sharpe: number | null;
  pnl_mean: number | null;
  pnl_total: number | null;
  net_pips_per_year: number | null;
  max_dd: number | null;
};

export type RunsResponse = { db: string; missing?: boolean; runs: RunListItem[]; leaderboard: StrategySummaryRow[] };

export type PathRow = {
  id: number;
  market: string;
  market_label: string;
  seed: number;
  detail: number;
  status: Status;
  realism_score: number | null;
  n_fail: number | null;
  facts: Record<string, number | undefined>;
};

/** ジョブの成績のうち、パターンごとの統計を取れる指標。 */
export type MetricKey = "pnl" | "net_pips_per_year" | "sharpe" | "gross_sharpe" | "max_dd" | "trades_per_year" | "win_rate";

export type JobRow = Record<MetricKey, number | null> & {
  id: number;
  path_id: number;
  broker: string;
  strategy: string;
  strategy_label: string;
  timeframe: number;
  status: Status;
  note: string | null;
  bars_total: number | null;
  bars_done: number | null;
  n_trades: number | null;
  pnl_live: number | null;
  market_label: string;
  market: string;
  seed: number;
  detail: number;
};

/** 戦略の成績を束ねる単位。market が "ALL" の行は全市場の総合。 */
export type JobGroup = { broker: string; strategy: string; market: string; jobIds: number[] };

export type RunDetail = {
  run: {
    id: number;
    name: string;
    kind: string;
    status: Status;
    years: number;
    seeds_json: string;
    git_sha: string | null;
    proposal: string | null;
  };
  initialBalance: number;
  paths: PathRow[];
  jobs: JobRow[];
  groups: JobGroup[];
  progress: { finished: number; total: number };
};

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type EqPoint = { t: number; balance: number; equity: number };
export type Trade = {
  seq: number;
  side: number;
  entry_t: number;
  exit_t: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pips: number;
  cost_pips: number;
  reason: string;
  bars_held: number;
};
export type MarketEvent = { t: number; kind: string; direction: number | null; pips: number | null };

export type JobMetrics = {
  cagr: number;
  sharpe: number;
  gross_sharpe: number;
  max_dd: number;
  profit_factor: number | null;
  win_rate: number;
  trades_per_year: number;
  gross_pips_per_year: number;
  cost_pips_per_year: number;
  swap_total: number;
  hunted: number;
  exit_reasons: Record<string, number>;
};

export type JobInfo = {
  id: number;
  run_id: number;
  run_name: string;
  strategy_label: string;
  market_label: string;
  seed: number;
  broker: string;
  status: Status;
  timeframe: number;
  bars_total: number | null;
  bars_done: number | null;
  sim_t: number | null;
  t_start: number | null;
  balance: number | null;
  equity: number | null;
  n_trades: number | null;
  pnl: number | null;
  pnl_live: number | null;
  params: Record<string, unknown>;
  schedule: Schedule | null;
  metrics: JobMetrics | null;
};

export type JobResponse = {
  job: JobInfo;
  initialBalance: number;
  tf: number;
  candleTf: number;
  candles: Candle[];
  equity: EqPoint[];
  trades: Trade[];
  events: MarketEvent[];
};

export type Summary = {
  n: number;
  mean: number;
  median: number;
  total: number;
  std: number;
  min: number;
  max: number;
  p10: number;
  p90: number;
  positive_ratio: number;
  t: number;
};

export type PatternDef = { market: string; label?: string | null; seeds: number[] };

/** case_defs.definition_json(動作ケースとテストケースで使う項目が違う)。 */
export type CaseDefinition = {
  // 動作ケース
  strategy?: string;
  params?: Record<string, unknown>;
  timeframe?: number | null;
  schedule?: Schedule;
  broker?: string | { preset: string };
  // テストケース
  behavior?: string;
  period?: { years?: number; start?: string; end?: string };
  patterns?: PatternDef[];
};

export type CaseDef = { id: string; kind: "behavior" | "test"; name: string; hash: string; def: CaseDefinition };
export type CaseRun = {
  id: number;
  test_id: string;
  run_id: number | null;
  status: Status;
  test_hash: string;
  summary: {
    all: Record<string, Summary> | null;
    by_market: Record<string, Record<string, Summary>>;
    verdict?: { verdict: string };
  } | null;
};
export type CasesResponse = {
  behaviors: CaseDef[];
  tests: (CaseDef & { runs: CaseRun[]; current: boolean })[];
};

// ---- 検証(fxsim/verification.py の出力) ----
export type VerdictLabel = "合格" | "条件付き" | "不合格";
export type VerdictCheck = { key: string; label: string; ok: boolean; detail: string };
export type VerdictGroup = Summary & { label: string; null_mean?: number };
export type Verdict = {
  verdict: VerdictLabel;
  checks: VerdictCheck[];
  groups: Record<string, VerdictGroup>;
  paired: Summary | null;
  yearly: (Summary & { year: string; coverage: number })[];
  model: string;
};

export type FanLine = { jobId: number; market: string; seed: number; real: boolean; points: { t: number; v: number }[] };
export type FanPoint = { t: number; median: number; p10: number; p90: number };
export type FanGroup = {
  key: string;
  label: string;
  lines: FanLine[];
  band: FanPoint[];
  nullMedian: { t: number; v: number }[];
};
export type YearRow = { label: string; market: string; seed: number; real: boolean; values: Record<string, number> };

export type CaseDetail = {
  test: CaseDef;
  behavior: CaseDef | null;
  runId: number | null;
  runStatus: Status | null;
  progress: { finished: number; total: number };
  verdict: Verdict | null;
  fan: FanGroup[];
  years: string[];
  yearRows: YearRow[];
};
