import Database from "better-sqlite3";
import path from "node:path";
import fs from "node:fs";

// 既定はリポジトリ直下の data/fxsim.sqlite。FXSIM_DB で差し替えられる。
export const DB_PATH = process.env.FXSIM_DB ?? path.resolve(process.cwd(), "..", "data", "fxsim.sqlite");

let db: Database.Database | null = null;

export function getDb(): Database.Database | null {
  if (db) return db;
  if (!fs.existsSync(/*turbopackIgnore: true*/ DB_PATH)) return null;
  // シミュレーション(Python)が書いている最中でも読めるよう、読み取り専用で開く(WAL)
  db = new Database(/*turbopackIgnore: true*/ DB_PATH, { readonly: true, fileMustExist: true });
  db.pragma("busy_timeout = 5000");
  return db;
}

export type Row = Record<string, any>;

export function all(sql: string, ...params: unknown[]): Row[] {
  const d = getDb();
  if (!d) return [];
  return d.prepare(sql).all(...params) as Row[];
}

export function one(sql: string, ...params: unknown[]): Row | undefined {
  const d = getDb();
  if (!d) return undefined;
  return d.prepare(sql).get(...params) as Row | undefined;
}

export function parseJson<T = any>(s: string | null | undefined, fallback: T): T {
  if (!s) return fallback;
  try {
    return JSON.parse(s.replace(/\bNaN\b/g, "null").replace(/-?\bInfinity\b/g, "null")) as T;
  } catch {
    return fallback;
  }
}
