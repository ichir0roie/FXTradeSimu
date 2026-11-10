import { NextResponse } from "next/server";
import { listRuns, strategyRanking } from "@/lib/queries";
import { DB_PATH, getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!getDb()) return NextResponse.json({ db: DB_PATH, missing: true, runs: [], leaderboard: [] });
  return NextResponse.json({ db: DB_PATH, runs: listRuns(), leaderboard: strategyRanking() });
}
