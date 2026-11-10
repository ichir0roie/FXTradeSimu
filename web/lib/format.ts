import type { Schedule } from "./types";

type N = number | null | undefined;
const ok = (x: N): x is number => x !== null && x !== undefined && Number.isFinite(x);

export function fmt(x: N, digits = 2): string {
  if (!ok(x)) return "–";
  if (Math.abs(x) >= 10000) return Math.round(x).toLocaleString("ja-JP");
  return x.toFixed(digits);
}

/** 円。プラスには + を付ける。 */
export function yen(x: N): string {
  if (!ok(x)) return "–";
  const s = Math.round(x).toLocaleString("ja-JP");
  return x > 0 ? `+${s}` : s;
}

export function pct(x: N, digits = 1): string {
  return ok(x) ? `${(x * 100).toFixed(digits)}%` : "–";
}

/** UNIX 秒 → "YYYY-MM-DD HH:MM"(UTC)。 */
export function date(sec: N): string {
  return ok(sec) && sec > 0 ? new Date(sec * 1000).toISOString().slice(0, 16).replace("T", " ") : "–";
}

/** 正負の色のクラス。 */
export function tone(x: N): string {
  if (!ok(x) || x === 0) return "";
  return x > 0 ? "pos" : "neg";
}

/** t 値の色: |t| ≥ 2 のときだけ正負の色、それ以外は控えめに。 */
export function tTone(t: N): string {
  return ok(t) && Math.abs(t) >= 2 ? tone(t) : "muted";
}

/** 分 → "M5" / "H1" / "H4" / "D1"。 */
export function tfLabel(minutes: N): string {
  if (!ok(minutes)) return "既定";
  if (minutes >= 1440) return `D${minutes / 1440}`;
  if (minutes >= 60) return `H${minutes / 60}`;
  return `M${minutes}`;
}

/** 取引時間 "07:00–21:00 UTC" / "終日"。 */
export function hoursLabel(s: Pick<Schedule, "hours" | "tz"> | null | undefined): string {
  if (!s?.hours?.length) return "終日";
  return `${s.hours.map(([a, b]) => `${a}–${b}`).join(", ")} ${s.tz}`;
}

export function outsideLabel(outside: string | undefined): string {
  return outside === "close" ? "手仕舞い" : "持ち越し";
}

/** 費用を損益の向き(マイナス)で見せる。0 を "-0" にしない。 */
export function neg(x: number | null | undefined): number | null {
  if (x === null || x === undefined) return null;
  return x === 0 ? 0 : -x;
}
