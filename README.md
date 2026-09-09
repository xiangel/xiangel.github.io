# xiangel blog

基于 [AstroPaper](https://github.com/satnaing/astro-paper) 的个人博客，使用 Astro + GitHub Actions + Markdown 构建。

## 特性

- 精致排版，明暗主题切换
- Markdown 写文章，Git 版本管理
- Pagefind 全文搜索
- GitHub Actions 自动部署

## 本地开发

```bash
npm install
npm run dev
```

访问 http://localhost:4321

## 写文章

在 `src/content/posts/` 目录下新建 Markdown 文件：

```markdown
---
author: xiangel
pubDatetime: 2026-08-31T10:00:00Z
title: 文章标题
slug: post-slug
draft: false
tags:
  - 标签
description: 文章摘要
---

正文内容...
```

## 发布

```bash
git add .
git commit -m "post: 文章标题"
git push
```

推送到 `main` 分支后自动部署到 https://xiangel.github.io

## 首次部署

在 GitHub 仓库 **Settings → Pages** 中，将 Source 设置为 **GitHub Actions**。

## Giscus 评论

文章页已集成 [Giscus](https://giscus.app/) 评论（基于 GitHub Discussions）。

### 一次性配置

1. 打开仓库 **Settings → General → Features**，勾选 **Discussions**
2. 安装 [Giscus GitHub App](https://github.com/apps/giscus)
3. 在 Discussions 中确认存在 **Announcements** 分类（没有则新建）
4. 本地运行获取 categoryId：

```bash
npm run giscus:setup
```

5. 将输出的 `categoryId` 填入 `astro-paper.config.ts`：

```ts
giscus: {
  categoryId: "DIC_kwDOMu995s4XXXXXX",
}
```

或在 GitHub 仓库 **Settings → Secrets and variables → Actions → Variables** 中添加 `PUBLIC_GISCUS_CATEGORY_ID`，用于 CI 构建。
