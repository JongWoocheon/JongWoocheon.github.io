// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://jongwoocheon.github.io",
  integrations: [
    starlight({
      title: "个人学习笔记",
      description: "记录学习过程中的知识、实践与思考。",
      locales: {
        root: {
          label: "简体中文",
          lang: "zh-CN",
        },
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/JongWoocheon",
        },
      ],
      sidebar: [
        {
          label: "计算机科学",
          items: [{ autogenerate: { directory: "computer-science" } }],
        },
        {
          label: "人工智能",
          items: [{ autogenerate: { directory: "artificial-intelligence" } }],
        },
        {
          label: "工具与实践",
          items: [{ autogenerate: { directory: "tools" } }],
        },
      ],
    }),
  ],
});
