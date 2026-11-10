"use client";
import { useEffect, useRef, useState } from "react";

/** url を取得し、active の間は interval ミリ秒ごとに取り直す。 */
export function usePoll<T>(url: string, active: (d: T | null) => boolean, interval = 1000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeRef = useRef(active);
  activeRef.current = active;
  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        const j = (await r.json()) as T;
        if (stop) return;
        setData(j);
        setError(null);
        if (activeRef.current(j)) timer = setTimeout(tick, interval);
      } catch (e) {
        if (stop) return;
        setError(String(e));
        timer = setTimeout(tick, interval * 3);
      }
    };
    tick();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  }, [url, interval]);
  return { data, error };
}
