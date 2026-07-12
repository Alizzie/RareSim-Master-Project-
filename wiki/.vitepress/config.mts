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
          { text: 'Installation', link: '/GettingStarted_Installation' },
          { text: 'Quick Start', link: '/GettingStarted_QuickStart' },
        ]
      },
      {
        text: 'Project Overview',
        items: [
          { text: 'Overview', link: '/ProjectOverview_Overview' },
          { text: 'Configuration', link: '/ProjectOverview_Configuration' },
          { text: 'Output', link: '/ProjectOverview_Output' },
        ]
      },
      {
        text: 'System',
        items: [
          { text: 'Architecture Design', link: '/System_ArchitectureDesign' },
          { text: 'Web Interface', link: '/System_WebInterface' },
        ]
      },
      {
        text: 'Artifacts',
        items: [
          { text: 'Shared Overview', link: '/Artifacts_SharedOverview' },
          { text: 'Full Workflow', link: '/Artifacts_FullWorkflow' },
          { text: 'Raw Sources & Ontology Loading', link: '/Artifacts_RawSourcesAndOntologyLoading' },
          { text: 'Ontology Overview', link: '/Artifacts_OntologyOverview' },
          { text: 'HPO', link: '/Artifacts_OntologyHPO' },
          { text: 'ORPHA', link: '/Artifacts_OntologyORPHA' },
          { text: 'OMIM', link: '/Artifacts_OntologyOMIM' },
          { text: 'Disease Profile Construction', link: '/Artifacts_DiseaseProfileConstruction' },
          { text: 'Disease ID Normalization & Mapping', link: '/Artifacts_DiseaseIDNormalizationAndMapping' },
          { text: 'File Reference & Runtime Loading', link: '/Artifacts_FileReferenceAndRuntimeLoading' },
        ]
      },
      {
        text: 'Similarity Methods',
        items: [
          { text: 'Embedding Methods', link: '/SimilarityMethods_Embedding' },
          { text: 'LLM', link: '/SimilarityMethods_LLM' },
        ]
      },
      {
        text: 'Evaluation',
        items: [
          { text: 'Workflow Overview', link: '/Evaluation_WorkflowOverview' },
          { text: 'Dataset Available', link: '/Evaluation_DatasetAvailable' },
          { text: 'Dataset Format', link: '/Evaluation_DatasetFormat' },
          { text: 'Dataset Adding', link: '/Evaluation_DatasetAdding' },
          { text: 'Cache Format', link: '/Evaluation_CacheFormat' },
          { text: 'Batch Runners & Shared Utilities', link: '/Evaluation_BatchRunnersAndSharedUtilities' },
          { text: 'Evaluator & Metrics', link: '/Evaluation_EvaluatorAndMetrics' },
          { text: 'Adding a New Evaluation Method', link: '/Evaluation_AddingMethod' },
          { text: 'Validation Tools Overview', link: '/Evaluation_ValidationToolsOverview' },
          { text: 'Validation: LIRICAL', link: '/Evaluation_ValidationLIRICAL' },
          { text: 'Validation: Phenomizer', link: '/Evaluation_ValidationPhenomiser' },
          { text: 'Validation: PhenoBrain', link: '/Evaluation_ValidationPhenoBrain' },
          { text: 'Validation: DX29 Search', link: '/Evaluation_ValidationDX29Search' },
          { text: 'Validation: DX29 Phrank', link: '/Evaluation_ValidationDX29Phrank' },
          { text: 'Validation Output', link: '/Evaluation_ValidationOutput' },
        ]
      },
    ]
  }
})
