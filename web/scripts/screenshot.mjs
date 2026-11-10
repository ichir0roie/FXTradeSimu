// 画面のスクリーンショットを撮る。web サーバーを起動した状態で:
//   node scripts/screenshot.mjs [http://localhost:3000] [../docs/screenshots] [runId] [jobId]
// runId / jobId を省くと、最新の実験と、その中のチャート付きジョブを選ぶ。
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
let playwright;
try {
  playwright = require("playwright");
} catch {
  console.error("playwright が見つかりません。`npm i -D playwright && npx playwright install chromium` を実行してください。");
  process.exit(1);
}

const base = process.argv[2] ?? "http://localhost:3000";
const out = path.resolve(process.argv[3] ?? "../docs/screenshots");
fs.mkdirSync(out, { recursive: true });

const runs = await (await fetch(`${base}/api/runs`)).json();
const runId = Number(process.argv[4] ?? runs.runs?.[0]?.id);
const detail = await (await fetch(`${base}/api/runs/${runId}`)).json();
const pick =
  detail.jobs.find((j) => j.detail && j.status !== "skipped" && j.strategy !== "random_entry" && j.n_trades > 0) ??
  detail.jobs.find((j) => j.detail && j.status !== "skipped");
const jobId = Number(process.argv[5] ?? pick?.id);

const browser = await playwright.chromium.launch();
const shots = [
  ["01-runs.png", "/", 900],
  [`02-run-${runId}.png`, `/runs/${runId}`, 1600],
  [`03-job-${jobId}.png`, `/jobs/${jobId}`, 1400],
];
for (const scheme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: scheme });
  for (const [name, url, h] of shots) {
    await page.setViewportSize({ width: 1440, height: h });
    await page.goto(base + url, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    const file = path.join(out, name.replace(".png", `-${scheme}.png`));
    await page.screenshot({ path: file, fullPage: true });
    console.log(file);
  }
  await page.close();
}
await browser.close();
