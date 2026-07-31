import { h, nextTick } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
import { createMermaidRenderer } from 'vitepress-mermaid-renderer'

export default {
  extends: DefaultTheme,
  Layout: () => {
    const { isDark } = useData()
    const initMermaid = () => createMermaidRenderer({ theme: isDark.value ? 'dark' : 'default' })
    nextTick(() => initMermaid())
    return h(DefaultTheme.Layout)
  },
} satisfies Theme
