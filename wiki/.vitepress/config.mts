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
          { text: 'Ontology Overview', link: '/artifacts/ontology-overview' },
          { text: 'HPO', link: '/artifacts/ontology-hpo' },
          { text: 'ORPHA', link: '/artifacts/ontology-orpha' },
          { text: 'OMIM', link: '/artifacts/ontology-omim' },
          { text: 'Disease Profile Construction', link: '/artifacts/disease-profile-construction' },
          { text: 'Disease ID Normalization & Mapping', link: '/artifacts/disease-id-normalization-and-mapping' },
          { text: 'File Reference & Runtime Loading', link: '/artifacts/file-reference-and-runtime-loading' },
        ]
      },
      {
        text: 'Similarity Methods',
        items: [
          { text: 'Embedding Methods', link: '/similarity-methods/embedding' },
          { text: 'LLM', link: '/similarity-methods/llm' },
        ]
      },
      {
        text: 'Evaluation',
        items: [
          { text: 'Workflow Overview', link: '/evaluation/workflow-overview' },
          { text: 'Dataset Available', link: '/evaluation/dataset-available' },
          { text: 'Dataset Format', link: '/evaluation/dataset-format' },
          { text: 'Dataset Adding', link: '/evaluation/dataset-adding' },
          { text: 'Cache Format', link: '/evaluation/cache-format' },
          { text: 'Batch Runners & Shared Utilities', link: '/evaluation/batch-runners-and-shared-utilities' },
          { text: 'Evaluator & Metrics', link: '/evaluation/evaluator-and-metrics' },
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
