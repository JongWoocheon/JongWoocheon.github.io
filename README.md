# 个人学习笔记

使用 [Astro](https://astro.build/) 与 [Starlight](https://starlight.astro.build/) 构建的个人学习笔记站。站点采用纯静态预渲染，并通过 GitHub Actions 发布到 GitHub Pages。

## 本地开发

环境要求：

- Node.js 22.12.0 或更高版本
- pnpm 11.9.0

安装依赖并启动开发服务器：

```powershell
pnpm install --frozen-lockfile
pnpm dev
```

常用命令：

| 命令           | 用途                              |
| -------------- | --------------------------------- |
| `pnpm dev`     | 启动本地开发服务器                |
| `pnpm check`   | 检查 Astro、TypeScript 和内容集合 |
| `pnpm build`   | 构建生产版本到 `dist/`            |
| `pnpm preview` | 本地预览生产构建                  |

## 添加笔记

笔记位于 `src/content/docs/`，按主题目录组织。新页面至少包含：

```yaml
---
title: 页面标题
description: 页面摘要
sidebar:
  order: 1
---
```

侧栏会按主题目录自动生成。使用 `sidebar.order` 调整同一主题内的页面顺序。

## 部署

推送到 `main` 分支会触发 `.github/workflows/deploy.yml`。首次发布前，需要在 GitHub 仓库的 **Settings → Pages** 中将 Source 设置为 **GitHub Actions**。

当前部署配置：

- GitHub 用户名：`JongWoocheon`
- 仓库名：`JongWoocheon.github.io`
- 站点地址：`https://jongwoocheon.github.io/`

该仓库使用 GitHub Pages 用户主页地址，因此不配置 Astro `base`。如果以后迁移到普通项目仓库，需要在 `astro.config.mjs` 中增加与仓库名一致的 `base`。
