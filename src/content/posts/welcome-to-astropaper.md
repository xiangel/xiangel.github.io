---
author: xiangel
pubDatetime: 2026-08-31T10:00:00Z
title: 博客迁移至 AstroPaper
slug: welcome-to-astropaper
featured: true
draft: false
tags:
  - 博客
  - Astro
description: 使用 AstroPaper 主题重构博客，优化排版与阅读体验。
---

本站已从 Hexo 静态产物仓库，升级为 **Astro 源码仓库**，并采用 [AstroPaper](https://github.com/satnaing/astro-paper) 主题。

## 新工作流

1. 在 `src/content/posts/` 下新建 `.md` 文件
2. `git commit` 并 `git push` 到 `main` 分支
3. GitHub Actions 自动构建并发布到 GitHub Pages

## 技术栈

- **框架**：Astro + AstroPaper
- **样式**：Tailwind CSS
- **内容**：Markdown
- **部署**：GitHub Pages + GitHub Actions
- **搜索**：Pagefind

写博客，从此只需关注内容本身。
