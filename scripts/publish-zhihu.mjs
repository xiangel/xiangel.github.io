#!/usr/bin/env node
/**
 * Semi-automatic Zhihu article publisher (Playwright).
 *
 * Zhihu does NOT expose a public write API for individual creators.
 * This script automates the web editor after a one-time manual login.
 *
 * Setup:
 *   npm install -D playwright
 *   npx playwright install chromium
 *
 * Usage:
 *   node scripts/publish-zhihu.mjs --login
 *   node scripts/publish-zhihu.mjs kv-cache-paged-attention-and-prefix-caching --dry-run
 *   node scripts/publish-zhihu.mjs kv-cache-paged-attention-and-prefix-caching --publish
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const ROOT = resolve(__dirname, "..");
const AUTH_PATH = join(ROOT, ".secrets/zhihu-auth.json");
const EXPORT_ROOT = join(ROOT, "exports/zhihu");
const WRITE_URL = "https://zhuanlan.zhihu.com/write";

const args = process.argv.slice(2);
const isLogin = args.includes("--login");
const isDryRun = args.includes("--dry-run");
const isPublish = args.includes("--publish");
const slug = args.find(a => !a.startsWith("--"));

function usage() {
  console.log(`
知乎半自动发布工具

  首次登录（会打开浏览器，请手动完成登录/验证码）:
    npm run publish:zhihu -- --login

  预览填充效果（不点发布，保存截图）:
    npm run publish:zhihu -- <slug> --dry-run

  正式发布:
    npm run publish:zhihu -- <slug> --publish

示例:
    npm run publish:zhihu -- kv-cache-paged-attention-and-prefix-caching --dry-run
`);
}

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    console.error(
      "缺少 playwright。请先运行:\n  npm install -D playwright\n  npx playwright install chromium"
    );
    process.exit(1);
  }
}

function loadBundle(slugName) {
  const dir = join(EXPORT_ROOT, slugName);
  const mdPath = join(dir, "article.md");
  const htmlPath = join(dir, "article.html");

  if (!existsSync(mdPath)) {
    console.error(`未找到导出文件: ${mdPath}\n请先运行: npm run export:zhihu -- ${slugName}`);
    process.exit(1);
  }

  const md = readFileSync(mdPath, "utf8");
  const titleMatch = md.match(/^#\s+(.+)\n/);
  const title = titleMatch?.[1]?.trim() ?? slugName;
  const bodyMd = md.replace(/^#\s+.+\n+/, "").trim();
  const html = existsSync(htmlPath) ? readFileSync(htmlPath, "utf8") : "";

  return { dir, title, bodyMd, html };
}

async function loginFlow(chromium) {
  mkdirSync(join(ROOT, ".secrets"), { recursive: true });

  const browser = await chromium.launch({ headless: false, slowMo: 80 });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    locale: "zh-CN",
  });
  const page = await context.newPage();

  console.log("请在打开的浏览器中登录知乎，完成后回到终端按 Enter …");
  await page.goto("https://www.zhihu.com/signin", { waitUntil: "domcontentloaded" });

  await new Promise(resolve => {
    process.stdin.resume();
    process.stdin.once("data", () => resolve());
  });

  await context.storageState({ path: AUTH_PATH });
  console.log(`登录态已保存: ${AUTH_PATH}`);
  await browser.close();
}

async function pasteHtml(page, html) {
  const bodyHtml = html.match(/<body[^>]*>([\s\S]*)<\/body>/i)?.[1] ?? html;
  const cleaned = bodyHtml
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<div class="footer">[\s\S]*$/i, "")
    .replace(/<hr\s*\/?>\s*$/i, "");

  await page.evaluate(async content => {
    const htmlBlob = new Blob([content], { type: "text/html" });
    const textBlob = new Blob([content.replace(/<[^>]+>/g, "")], { type: "text/plain" });
    // @ts-ignore
    const item = new ClipboardItem({ "text/html": htmlBlob, "text/plain": textBlob });
    await navigator.clipboard.write([item]);
  }, cleaned);

  const editor = page.locator(".DraftEditor-root, .public-DraftEditor-content, [contenteditable='true']").first();
  await editor.click({ timeout: 15000 });
  await page.keyboard.press(process.platform === "darwin" ? "Meta+V" : "Control+V");
  await page.waitForTimeout(2500);
}

async function fillTitle(page, title) {
  const titleInput = page.locator(
    'textarea[placeholder*="标题"], input[placeholder*="标题"], .WriteIndex-titleInput textarea, .WriteIndex-titleInput input'
  ).first();
  await titleInput.waitFor({ state: "visible", timeout: 15000 });
  await titleInput.fill("");
  await titleInput.fill(title);
}

async function publishArticle(slugName, { dryRun }) {
  if (!existsSync(AUTH_PATH)) {
    console.error(`未找到登录态 ${AUTH_PATH}\n请先运行: npm run publish:zhihu -- --login`);
    process.exit(1);
  }

  const { chromium } = await loadPlaywright();
  const { dir, title, html } = loadBundle(slugName);

  const browser = await chromium.launch({ headless: false, slowMo: 50 });
  const context = await browser.newContext({
    storageState: AUTH_PATH,
    viewport: { width: 1280, height: 900 },
    locale: "zh-CN",
  });
  const page = await context.newPage();

  try {
    console.log(`打开发文页: ${WRITE_URL}`);
    await page.goto(WRITE_URL, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForTimeout(2000);

    console.log(`填写标题: ${title}`);
    await fillTitle(page, title);

    console.log("粘贴正文（来自 article.html）…");
    if (!html) throw new Error("article.html 不存在，请重新 export:zhihu");
    await pasteHtml(page, html);

    const screenshotPath = join(dir, dryRun ? "zhihu-dry-run.png" : "zhihu-before-publish.png");
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`截图已保存: ${screenshotPath}`);

    if (dryRun) {
      console.log("Dry-run 完成：未点击发布。请检查截图与浏览器窗口。");
      console.log("确认无误后运行: npm run publish:zhihu --", slugName, "--publish");
      await page.waitForTimeout(15000);
      return;
    }

    const publishBtn = page.locator('button:has-text("发布")').last();
    await publishBtn.waitFor({ state: "visible", timeout: 10000 });
    await publishBtn.click();

    await page.waitForURL(/zhuanlan\.zhihu\.com\/p\//, { timeout: 30000 }).catch(() => {});
    const url = page.url();
    if (/zhuanlan\.zhihu\.com\/p\//.test(url)) {
      writeFileSync(join(dir, "published-url.txt"), `${url}\n`);
      console.log(`✅ 发布成功: ${url}`);
    } else {
      console.log("已点击发布，但未检测到文章 URL。请在浏览器中确认是否成功。");
    }

    await context.storageState({ path: AUTH_PATH });
  } catch (error) {
    const errorShot = join(dir, "zhihu-error.png");
    await page.screenshot({ path: errorShot, fullPage: true }).catch(() => {});
    console.error(`❌ 发布失败: ${error.message}`);
    console.error(`错误截图: ${errorShot}`);
    throw error;
  } finally {
    await browser.close();
  }
}

async function main() {
  if (isLogin) {
    const { chromium } = await loadPlaywright();
    await loginFlow(chromium);
    return;
  }

  if (!slug || (!isDryRun && !isPublish)) {
    usage();
    process.exit(slug ? 1 : 0);
  }

  await publishArticle(slug, { dryRun: isDryRun });
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
