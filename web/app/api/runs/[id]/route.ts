import { NextResponse } from "next/server";
import { runDetail } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const d = runDetail(Number(id));
  if (!d) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(d);
}
