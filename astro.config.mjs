// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightThemeNext from "starlight-theme-next";
import { unified } from "@astrojs/markdown-remark";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

// https://astro.build/config
export default defineConfig({
  site: "https://jongwoocheon.github.io",
  devToolbar: {
    enabled: false,
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [
    starlight({
      title: "🧩Openote",
      description: "记录学习过程中的知识、实践与思考。",
      plugins: [starlightThemeNext()],
      customCss: ["./src/styles/custom.css"],
      components: {
        Sidebar: "./src/components/sidebar/Sidebar.astro",
      },
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
          label: "深度学习",
          items: [
            {
              label: "预备知识",
              collapsed: false,
              items: [
                "deep-learning",
              ],
            },
          ],
        },
      ],
    }),
  ],
});
