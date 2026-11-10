export function Progress({ value }: { value: number | null | undefined }) {
  const v = Math.max(0, Math.min(1, Number(value ?? 0)));
  return (
    <div className="bar" title={`${(v * 100).toFixed(1)}%`}>
      <i style={{ width: `${v * 100}%` }} />
    </div>
  );
}

export function Status({ s }: { s: string }) {
  const label: Record<string, string> = { running: "実行中", done: "完了", failed: "失敗", queued: "待機", skipped: "対象外" };
  return <span className={`badge ${s}`}>{label[s] ?? s}</span>;
}
