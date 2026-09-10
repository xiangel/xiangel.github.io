# 知乎发布说明

> 本文件仅供发布参考，不要粘贴到知乎正文。

## 建议标题

KV Cache 详解：从 PagedAttention 到 ChunkAttention

## 建议话题 / 标签

LLM、推理系统、KV-Cache

## 建议封面图

https://xiangel.github.io/assets/posts/kv-cache/diagram-prefill-decode.png

## 发布步骤

### 方式 A：半自动脚本（推荐）

知乎**没有**对个人开放的发文 API，无法在云端代你直接发布。仓库提供了 Playwright 半自动脚本：

```bash
# 1. 安装依赖（只需一次）
npm install -D playwright
npx playwright install chromium

# 2. 首次登录（打开浏览器，手动完成登录/验证码，回终端按 Enter）
npm run publish:zhihu -- --login

# 3. 预览填充（不点发布，生成 zhihu-dry-run.png）
npm run publish:zhihu -- kv-cache-paged-attention-and-prefix-caching --dry-run

# 4. 确认无误后正式发布
npm run publish:zhihu -- kv-cache-paged-attention-and-prefix-caching --publish
```

登录态保存在 `.secrets/zhihu-auth.json`（已 gitignore，不会提交）。

### 方式 B：手动复制粘贴

1. 打开 `article.html`，浏览器全选复制（Cmd/Ctrl+A → Cmd/Ctrl+C）。
2. 进入知乎「写文章」，直接粘贴到正文编辑器（保留标题、图片、代码块格式）。
3. 若图片未自动加载：在知乎编辑器里逐张上传 `public/assets/posts/kv-cache/` 下的 PNG。
4. 文末保留「原文链接」便于读者跳转博客。
5. 预览无误后发布。

## 备选方案（Markdown）

- 使用 `article.md`：图片链接已是公网 URL，可用 Markdown 编辑器或第三方导入工具。
- 知乎对 Markdown 支持有限，**优先推荐 HTML 复制粘贴**。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `article.md` | 纯 Markdown 正文，图片/链接已转绝对 URL |
| `article.html` | 浏览器复制粘贴到知乎编辑器 |
| `publish-notes.md` | 本说明 |

