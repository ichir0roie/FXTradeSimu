import { all, one, parseJson } from "../db";
import type { CaseDef, CaseRun, CasesResponse } from "../types";

export function casesOverview(): CasesResponse {
  if (!one(`SELECT name FROM sqlite_master WHERE type='table' AND name='case_defs'`)) return { behaviors: [], tests: [] };
  const defs = all(`SELECT id, kind, name, hash, definition_json FROM case_defs ORDER BY id`).map(
    ({ definition_json, ...d }) => ({ ...d, def: parseJson(definition_json, {}) }),
  ) as CaseDef[];
  const runs = all(
    `SELECT id, test_id, run_id, status, test_hash, summary_json FROM case_runs ORDER BY id DESC`,
  ).map(({ summary_json, ...r }) => ({ ...r, summary: parseJson(summary_json, null) })) as CaseRun[];
  const tests = defs
    .filter((d) => d.kind === "test")
    .map((t) => {
      const mine = runs.filter((r) => r.test_id === t.id);
      return { ...t, runs: mine, current: mine[0]?.test_hash === t.hash };
    });
  return { behaviors: defs.filter((d) => d.kind === "behavior"), tests };
}
