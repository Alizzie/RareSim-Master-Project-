import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "RareSim Wiki",
  description: "Documentation for the RareSim project",
  base: "/RareSim-Master-Project-/",
  themeConfig: {
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Installation', link: '/getting-started/installation' },
          { text: 'Quick Start', link: '/getting-started/quick-start' },
        ]
      },
      {
        text: 'Project Overview',
        items: [
          { text: 'Overview', link: '/project-overview/overview' },
          { text: 'Configuration', link: '/project-overview/configuration' },
          { text: 'Output', link: '/project-overview/output' },
        ]
      },
      {
        text: 'System',
        items: [
          { text: 'Architecture Design', link: '/system/architecture-design' },
          { text: 'Web Interface', link: '/system/web-interface' },
        ]
      },
      {
        text: 'Artifacts',
        items: [
          { text: 'Shared Overview', link: '/artifacts/shared-overview' },
          { text: 'Full Workflow', link: '/artifacts/full-workflow' },
          { text: 'Raw Sources & Ontology Loading', link: '/artifacts/raw-sources-and-ontology-loading' },
          { text: 'Disease ID Normalization & Mapping', link: '/artifacts/disease-id-normalization-and-mapping' },
          { text: 'Disease Profile Construction', link: '/artifacts/disease-profile-construction' },
          { text: 'File Reference & Runtime Loading', link: '/artifacts/file-reference-and-runtime-loading' },
        ]
      },
      {
        text: 'Similarity Methods',
        items: [
          { text: 'Embedding Methods', link: '/similarity-methods/embedding' },
          { text: 'LLM', link: '/similarity-methods/llm' },
          { text: 'Adding a new similarity method', link: '/similarity-methods/adding-new-method' },
        ]
      },
      {
        text: 'Evaluation',
        items: [
          { text: 'Workflow Overview', link: '/evaluation/workflow-overview' },
          { text: 'Dataset Format', link: '/evaluation/dataset-format' },
          { text: 'Dataset Available', link: '/evaluation/dataset-available' },
          { text: 'Dataset Adding', link: '/evaluation/dataset-adding' },
          { text: 'Batch Runners & Shared Utilities', link: '/evaluation/batch-runners-and-shared-utilities' },
          { text: 'Cache Format', link: '/evaluation/cache-format' },
          { text: 'Evaluator & Metrics', link: '/evaluation/evaluator-and-metrics' },
          { text: 'Visualizing Results', link: '/evaluation/visualizing-results' },
          { text: 'Adding a New Evaluation Method', link: '/evaluation/adding-method' },
        ]
      },
      {
        text: 'Validation',
        items: [
          { text: 'Tools Overview', link: '/validation/tools-overview' },
          { text: 'LIRICAL', link: '/validation/lirical' },
          { text: 'Phenomizer', link: '/validation/phenomiser' },
          { text: 'PhenoBrain', link: '/validation/phenobrain' },
          { text: 'DX29 Search', link: '/validation/dx29-search' },
          { text: 'DX29 Phrank', link: '/validation/dx29-phrank' },
          { text: 'Validation Output', link: '/validation/output' },
        ]
      },
    ]
  }
})
