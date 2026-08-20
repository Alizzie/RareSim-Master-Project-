import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "RareSim Wiki",
  description: "Documentation for the RareSim project",
  base: process.env.VITEPRESS_BASE || "/",
  outDir: "../public",
  themeConfig: {
    sidebar: [
      {
        text: 'Project Overview',
        items: [
          { text: 'Overview', link: '/project-overview/overview' },
          { text: 'Configuration', link: '/project-overview/configuration' },
          { text: 'Output', link: '/project-overview/output' },
        ]
      },
      {
        text: 'Getting Started',
        items: [
          { text: 'Installation', link: '/getting-started/installation' },
          { text: 'Quick Start', link: '/getting-started/quick-start' },
        ]
      },
      {
        text: 'System',
        items: [
          { text: 'Architecture Design', link: '/system/architecture-design' },
          { text: 'Web Interface', link: '/system/web-interface' },
          { text: 'Web Interface (Demo)', link: '/system/web-interface-demo' },
          { text: 'CLI', link: '/system/cli' },
          { text: 'HPO Extraction Pipeline', link: '/system/hpo-extraction-pipeline' },
          { text: 'Data & Storage Lifecycle', link: '/system/data-and-storage-lifecycle' },
          { text: 'Deployment & External Dependencies', link: '/system/deployment-and-external-dependencies' },
          { text: 'Implementation & Libraries', link: '/system/implementation-and-libraries' },
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
          { text: 'Patient Profile Construction', link: '/artifacts/patient-profile-construction' },
          { text: 'File Reference & Runtime Loading', link: '/artifacts/file-reference-and-runtime-loading' },

        ]
      },
      {
        text: 'Similarity Methods',
        items: [
          { text: 'Methods Overview', link: '/similarity-methods/overview' },
          { text: 'TF-IDF', link: '/similarity-methods/tfidf-methods' },
          { text: 'Semantic Methods', link: '/similarity-methods/semantic-methods' },
          { text: 'Set-Based Methods', link: '/similarity-methods/set-based-methods' },
          { text: 'HPO2Vec', link: '/similarity-methods/hpo2vec' },
          { text: 'Denoising Autoencoder', link: '/similarity-methods/denoising-autoencoder' },
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
          { text: 'Available Datasets', link: '/evaluation/dataset-available' },
          { text: 'Adding a Dataset', link: '/evaluation/dataset-adding' },
          { text: 'Benchmark Dataset Standardization', link: '/evaluation/benchmark-dataset-standardization' },
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
          { text: `PhenoBrain Local`, link: '/validation/phenobrain-local' },
          { text: 'DX29 Search', link: '/validation/dx29-search' },
          { text: 'DX29 Phrank', link: '/validation/dx29-phrank' },
          { text: 'Validation Output', link: '/validation/output' },
        ]
      },
    ]
  }
})
