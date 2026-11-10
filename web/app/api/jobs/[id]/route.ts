import { NextResponse } from "next/server";
import { jobDetail } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const u = new URL(req.url);
  const num = (k: string, d: number) => (u.searchParams.has(k) ? Number(u.searchParams.get(k)) : d);
  const d = jobDetail(Number(id), {
    candleFrom: num("candleFrom", 0),
    eqFrom: num("eqFrom", 0),
    tradeFrom: num("tradeFrom", 0),
    tf: u.searchParams.has("tf") ? Number(u.searchParams.get("tf")) : null,
  });
  if (!d) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(d);
}
