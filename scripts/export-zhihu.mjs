#!/usr/bin/env node
/**
 * Export blog posts for Zhihu publishing.
 *
 * Usage:
 *   node scripts/export-zhihu.mjs <slug-or-path>
 *   npm run export:zhihu -- kv-cache-paged-attention-and-prefix-caching
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const POSTS_DIR = join(ROOT, "src/content/posts");
const EXPORT_DIR = join(ROOT, "exports/zhihu");
const SITE_URL = "https://xiangel.github.io";

function loadSiteUrl() {
  try {
    const config = readFileSync(join(ROOT, "astro-paper.config.ts"), "utf8");
    const match = config.match(/url:\s*["']([^"']+)["']/);
    return match?.[1]?.replace(/\/$/, "") ?? SITE_URL;
  } catch {
    return SITE_URL;
  }
}

function parseFrontmatter(raw) {
  if (!raw.startsWith("---\n")) return { meta: {}, body: raw };
  const end = raw.indexOf("\n---\n", 4);
  if (end === -1) return { meta: {}, body: raw };
  const fm = raw.slice(4, end);
  const body = raw.slice(end + 5);
  const meta = {};
  const tags = [];
  let inTags = false;

  for (const line of fm.split("\n")) {
    if (/^tags:\s*$/.test(line)) {
      inTags = true;
      continue;
    }
    if (inTags) {
      const tagMatch = line.match(/^\s+-\s+(.+)$/);
      if (tagMatch) {
        tags.push(tagMatch[1].trim());
        continue;
      }
      inTags = false;
    }

    const m = line.match(/^(\w+):\s*(.+)$/);
    if (!m) continue;
    let value = m[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    meta[m[1]] = value;
  }

  if (tags.length) meta.tags = tags;
  return { meta, body };
}

function resolvePostPath(input) {
  if (input.endsWith(".md")) {
    return resolve(ROOT, input);
  }
  const direct = join(POSTS_DIR, `${input}.md`);
  if (basename(input) === input) return direct;
  return resolve(ROOT, input);
}

function transformForZhihu(body, siteUrl, slug) {
  let text = body;

  // Remove empty auto-TOC placeholder from AstroPaper.
  text = text.replace(/^## Table of contents\s*\n+/m, "");

  // Absolute image URLs (Zhihu needs publicly reachable images).
  text = text.replace(
    /!\[([^\]]*)\]\((\/assets\/[^)]+)\)/g,
    (_, alt, path) => `![${alt}](${siteUrl}${path})`
  );

  // Internal post links.
  text = text.replace(/\]\((\/posts\/[^)]+)\)/g, (_, path) => `](${siteUrl}${path})`);

  // Trim trailing whitespace.
  text = text.replace(/\s+$/, "") + "\n";

  text += `\n---\n\n**原文链接**：${siteUrl}/posts/${slug}/\n\n**系列上一篇**：[从 Transformer 出发来看推理系统](${siteUrl}/posts/from-causal-lm-to-inference-system/)\n`;

  return text;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

const MIME_BY_EXT = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  svg: "image/svg+xml",
};

function resolveLocalAssetPath(src) {
  const match = src.match(/\/assets\/(.+\.(?:png|jpe?g|gif|webp|svg))(?:[?#].*)?$/i);
  if (!match) return null;
  const localPath = join(ROOT, "public/assets", match[1]);
  return existsSync(localPath) ? localPath : null;
}

function embedImageAsDataUri(src) {
  const localPath = resolveLocalAssetPath(src);
  if (!localPath) return src;
  const ext = basename(localPath).split(".").pop()?.toLowerCase() ?? "png";
  const mime = MIME_BY_EXT[ext] ?? "application/octet-stream";
  const base64 = readFileSync(localPath).toString("base64");
  return `data:${mime};base64,${base64}`;
}

function collectImagePaths(body, siteUrl) {
  const paths = new Set();
  for (const match of body.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    const src = match[1];
    const localPath = resolveLocalAssetPath(src.startsWith("/") ? src : src.replace(siteUrl, ""));
    if (localPath) paths.add(localPath);
  }
  return [...paths];
}

function markdownToHtml(body, siteUrl, slug) {
  const lines = body.split("\n");
  const html = [];
  let inCode = false;
  let codeLang = "";
  let codeLines = [];
  let inTable = false;
  let tableRows = [];

  const flushTable = () => {
    if (!tableRows.length) return;
    html.push("<table>");
    tableRows.forEach((row, idx) => {
      const tag = idx === 1 && row.every(cell => /^:?-+:?$/.test(cell.trim()))
        ? null
        : idx === 0
          ? "th"
          : "td";
      if (tag === null) return;
      const rowTag = idx === 0 ? "thead" : idx === 2 ? "tbody" : null;
      if (rowTag === "thead") html.push("<thead>");
      if (rowTag === "tbody") html.push("<tbody>");
      html.push("<tr>");
      for (const cell of row) {
        html.push(`<${tag}>${inlineMarkdown(cell.trim())}</${tag}>`);
      }
      html.push("</tr>");
      if (idx === 0) html.push("</thead>");
    });
    if (tableRows.length > 2) html.push("</tbody>");
    html.push("</table>");
    tableRows = [];
    inTable = false;
  };

  let listType = null;
  let listItems = [];

  const flushList = () => {
    if (!listItems.length) return;
    html.push(`<${listType}>`);
    for (const item of listItems) {
      html.push(`<li>${inlineMarkdown(item)}</li>`);
    }
    html.push(`</${listType}>`);
    listItems = [];
    listType = null;
  };

  for (const line of lines) {
    if (line.startsWith("```")) {
      flushList();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
        codeLang = "";
      } else {
        flushTable();
        inCode = true;
        codeLang = line.slice(3).trim();
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (line.includes("|") && line.trim().startsWith("|")) {
      flushList();
      inTable = true;
      tableRows.push(
        line
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
      );
      continue;
    } else if (inTable) {
      flushTable();
    }

    const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (image) {
      flushList();
      const [, alt, src] = image;
      const embeddedSrc = embedImageAsDataUri(src);
      html.push(
        `<figure><img src="${embeddedSrc}" alt="${escapeHtml(alt)}" /><figcaption>${escapeHtml(alt)}</figcaption></figure>`
      );
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    if (line.startsWith("> ")) {
      flushList();
      html.push(`<blockquote><p>${inlineMarkdown(line.slice(2))}</p></blockquote>`);
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      if (listType && listType !== "ul") flushList();
      listType = "ul";
      listItems.push(line.replace(/^[-*]\s+/, ""));
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      if (listType && listType !== "ol") flushList();
      listType = "ol";
      listItems.push(line.replace(/^\d+\.\s+/, ""));
      continue;
    }

    flushList();

    if (!line.trim()) {
      html.push("");
      continue;
    }

    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  flushList();

  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushTable();

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>知乎发布稿</title>
  <style>
    body { max-width: 760px; margin: 40px auto; padding: 0 20px; font: 16px/1.8 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; color: #1f2329; }
    h1, h2, h3 { line-height: 1.4; margin: 1.6em 0 0.8em; }
    h1 { font-size: 28px; }
    h2 { font-size: 22px; border-left: 4px solid #0066ff; padding-left: 12px; }
    h3 { font-size: 18px; }
    p, blockquote, pre, figure, table, ul, ol { margin: 0.9em 0; }
    blockquote { padding: 12px 16px; background: #f6f8fa; border-left: 4px solid #c8cdd4; color: #4e5969; }
    pre { padding: 14px 16px; background: #f6f8fa; border-radius: 8px; overflow-x: auto; font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
    code { font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; background: #f2f3f5; padding: 0 4px; border-radius: 4px; }
    pre code { background: transparent; padding: 0; }
    figure { margin: 1.2em 0; text-align: center; }
    img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.06); }
    figcaption { margin-top: 8px; font-size: 13px; color: #86909c; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border: 1px solid #e5e6eb; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f7f8fa; }
    ul, ol { padding-left: 1.4em; }
    hr { border: none; border-top: 1px solid #e5e6eb; margin: 2em 0; }
    .footer { margin-top: 2.5em; padding-top: 1em; border-top: 1px solid #e5e6eb; color: #86909c; font-size: 14px; }
  </style>
</head>
<body>
${html.join("\n")}
<hr />
<div class="footer">
  <p>原文链接：<a href="${siteUrl}/posts/${slug}/">${siteUrl}/posts/${slug}/</a></p>
  <p>系列上一篇：<a href="${siteUrl}/posts/from-causal-lm-to-inference-system/">从 Transformer 出发来看推理系统</a></p>
</div>
</body>
</html>`;
}

function buildPublishNotes(meta, slug, siteUrl) {
  const tags = Array.isArray(meta.tags)
    ? meta.tags.join("、")
    : "LLM、推理系统、KV Cache";

  return `# 知乎发布说明

> 本文件仅供发布参考，不要粘贴到知乎正文。

## 建议标题

${meta.title ?? slug}

## 建议话题 / 标签

${tags}

## 建议封面图

${siteUrl}/assets/posts/kv-cache/diagram-prefill-decode.png

## 发布步骤

1. 打开 \`article.html\`（**图片已内嵌 base64**，不依赖外链）。
2. 浏览器全选复制（Cmd/Ctrl+A → Cmd/Ctrl+C）。
3. 进入知乎「写文章」，粘贴到正文编辑器。
4. 若粘贴后图片仍丢失：从同目录 \`images/\` 文件夹手动上传对应 PNG。
5. 文末保留「原文链接」便于读者跳转博客。
6. 预览无误后发布。

## 备选方案（Markdown）

- 使用 \`article.md\`：图片链接已是公网 URL，可用 Markdown 编辑器或第三方导入工具。
- 知乎对 Markdown 支持有限，**优先推荐 HTML 复制粘贴**。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| \`article.md\` | 纯 Markdown 正文，图片/链接已转绝对 URL |
| \`article.html\` | 浏览器复制粘贴到知乎编辑器 |
| \`publish-notes.md\` | 本说明 |

`;
}

function main() {
  const input = process.argv[2];
  if (!input) {
    console.error("Usage: node scripts/export-zhihu.mjs <slug-or-path>");
    process.exit(1);
  }

  const postPath = resolvePostPath(input);
  const raw = readFileSync(postPath, "utf8");
  const { meta, body } = parseFrontmatter(raw);
  const slug = meta.slug ?? basename(postPath, ".md");
  const siteUrl = loadSiteUrl();
  const transformed = transformForZhihu(body, siteUrl, slug);
  const outDir = join(EXPORT_DIR, slug);

  mkdirSync(outDir, { recursive: true });

  const mdPath = join(outDir, "article.md");
  const htmlPath = join(outDir, "article.html");
  const notesPath = join(outDir, "publish-notes.md");

  writeFileSync(mdPath, `# ${meta.title ?? slug}\n\n${transformed}`);
  writeFileSync(
    htmlPath,
    markdownToHtml(`# ${meta.title ?? slug}\n\n${transformed}`, siteUrl, slug)
  );
  writeFileSync(notesPath, buildPublishNotes(meta, slug, siteUrl));

  const imagesDir = join(outDir, "images");
  mkdirSync(imagesDir, { recursive: true });
  const imagePaths = collectImagePaths(transformed, siteUrl);
  for (const localPath of imagePaths) {
    copyFileSync(localPath, join(imagesDir, basename(localPath)));
  }

  console.log(`Exported Zhihu bundle to ${outDir}`);
  console.log(`  - ${mdPath}`);
  console.log(`  - ${htmlPath} (images embedded as base64)`);
  console.log(`  - ${notesPath}`);
  if (imagePaths.length) {
    console.log(`  - ${imagesDir}/ (${imagePaths.length} images for manual upload fallback)`);
  }
}

main();
