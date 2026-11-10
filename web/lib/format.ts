export function fmt(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  if (Math.abs(x) >= 10000) return Math.round(x).toLocaleString("ja-JP");
  return x.toFixed(digits);
}

export function yen(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  const s = Math.round(x).toLocaleString("ja-JP");
  return x > 0 ? `+${s}` : s;
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return `${(x * 100).toFixed(digits)}%`;
}

export function date(sec: number | null | undefined): string {
  if (!sec) return "–";
  return new Date(sec * 1000).toISOString().slice(0, 16).replace("T", " ");
}

export function tone(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x) || x === 0) return "";
  return x > 0 ? "pos" : "neg";
}
