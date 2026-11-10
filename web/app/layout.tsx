import "./globals.css";
import Link from "next/link";
import type { ReactNode } from "react";
import { DB_PATH } from "@/lib/db";

export const metadata = { title: "FXTradeSimu", description: "悪意ある相場のシミュレーション結果ビューア(ローカル専用)" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <header className="top">
          <Link href="/" className="brand">FXTradeSimu</Link>
          <span className="sub">悪意ある相場のシミュレーション</span>
          <nav style={{ display: "flex", gap: 14, marginLeft: 12 }}>
            <Link href="/">実験</Link>
            <Link href="/cases">テストケース</Link>
          </nav>
          <span className="db" title={DB_PATH}>{DB_PATH}</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
