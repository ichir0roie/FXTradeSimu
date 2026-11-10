import { parseJson, type Row } from "../db";

/** 実験の設定から初期資金を取る(古い run は engine が部分的なので既定値で補う)。 */
export function initialBalance(run: Row | undefined): number {
  const cfg = parseJson<{ engine?: { initial_balance?: number } }>(run?.config_json, {});
  return Number(cfg.engine?.initial_balance ?? 1_000_000);
}

/** 実行中は有効証拠金から、完了後は確定した損益を返す。 */
export function livePnl(job: Row, initial: number): number | null {
  if (job.status === "done") return job.pnl;
  return job.equity != null ? job.equity - initial : null;
}
