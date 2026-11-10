// パターンごとの値を横一列の点で並べる。橙のひし形 = 実データ、縦線 = 平均(実線)と中央値(破線)。
export type Pattern = { value: number | null; real: boolean; seed: number; market: string };

export function StripPlot({ patterns, lo, hi, mean, median, width = 240 }: {
  patterns: Pattern[]; lo: number; hi: number; mean?: number; median?: number; width?: number;
}) {
  const h = 22;
  const pad = 6;
  const span = hi - lo || 1;
  const x = (v: number) => pad + ((v - lo) / span) * (width - 2 * pad);
  return (
    <svg width={width} height={h} role="img" aria-label="パターンごとの値の分布">
      {lo < 0 && hi > 0 && <line x1={x(0)} x2={x(0)} y1={2} y2={h - 2} stroke="var(--border)" strokeWidth={1.5} />}
      {patterns.filter((p) => !p.real && p.value != null).map((p, i) => (
        <circle key={i} cx={x(p.value!)} cy={h / 2} r={3.2} fill="var(--dot)" fillOpacity={0.55}>
          <title>{`${p.market} seed${p.seed}: ${Math.round(p.value!).toLocaleString("ja-JP")}`}</title>
        </circle>
      ))}
      {mean != null && <line x1={x(mean)} x2={x(mean)} y1={3} y2={h - 3} stroke="var(--text)" strokeWidth={1.5} />}
      {median != null && (
        <line x1={x(median)} x2={x(median)} y1={3} y2={h - 3} stroke="var(--text)" strokeWidth={1} strokeDasharray="2 2" />
      )}
      {patterns.filter((p) => p.real && p.value != null).map((p, i) => {
        const cx = x(p.value!);
        return (
          <path key={`r${i}`} d={`M${cx} ${h / 2 - 5} L${cx + 5} ${h / 2} L${cx} ${h / 2 + 5} L${cx - 5} ${h / 2} Z`}
            fill="var(--real)" stroke="var(--panel)" strokeWidth={1}>
            <title>{`実データ: ${Math.round(p.value!).toLocaleString("ja-JP")}`}</title>
          </path>
        );
      })}
    </svg>
  );
}
