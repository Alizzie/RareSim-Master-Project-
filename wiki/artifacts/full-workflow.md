# Shared Artifact Workflow Figure

```mermaid
flowchart TD
    A["1. Download raw ontology/source files<br/>raresim/build/load_ontologies_to_local.py"] --> B["Raw files saved locally<br/>data/ontologies/"]

    B --> C["2. Build shared artifacts<br/>raresim/build/build_shared_artifacts.py"]

    C --> D["Load HPO ontology<br/>hpo.owl"]
    D --> D1["hpo_labels<br/>hpo_parents"]
    D1 --> D2["Compute HPO ancestors<br/>raresim/ontology/hpo_utils.py"]
    D2 --> D3["hpo_ancestors"]

    C --> E["Load phenotype annotation sources"]
    E --> E1["phenotype.hpoa<br/>hoom.owl<br/>en_product4_HPO.xml<br/>Monarch TSV.gz"]
    E1 --> E2["Raw phenotype records"]
    E2 --> EC["annotation_source_counts"]
    E2 --> E3["Merge phenotype records<br/>raresim/ontology/phenotype_merge.py"]
    E3 --> E4["Deduplicated positive records"]
    E3 --> EP["term_provenance"]
    E3 --> EN["negative_terms_by_disease"]

    C --> F["Load disease metadata"]
    F --> F1["ordo.owl<br/>mondo_rare.owl"]
    F1 --> F2["ORDO metadata<br/>MONDO metadata"]
    F2 --> F3["Build disease metadata index<br/>disease_metadata_index"]

    F2 --> G["Build ORPHA mapping index<br/>raresim/utils/mapping_utils.py"]
    G --> G1["orpha_mapping_index"]

    E4 --> H["Build canonical disease profiles<br/>raresim/ontology/disease_profiles.py"]
    EP --> H
    EN --> H
    D3 --> H
    D1 --> H
    F2 --> H
    G1 --> H

    H --> H1["canonical_disease_profiles<br/>alias_to_canonical"]
    H1 --> I["Expand alias profiles"]
    I --> I1["disease_profiles"]

    H1 --> J["Filter empty profiles"]
    I1 --> J
    J --> J1["canonical_filter_stats<br/>expanded_filter_stats"]

    C --> K["Build disease hierarchy"]
    K --> K1["load_ordo_parents<br/>ordo.owl"]
    K1 --> K2["disease_parents"]
    K2 --> K3["Build ordered ancestor chains<br/>raresim/ontology/disease_ancestors.py"]
    K3 --> K4["disease_ancestors"]

    H1 --> L["Compute term frequencies and IC<br/>raresim/ontology/ic.py"]
    L --> L1["term_frequencies<br/>information_content"]

    D1 --> M["Build example patient<br/>EXAMPLE_PATIENT"]
    D3 --> M
    M --> M1["example_patient"]

    D1 --> N["Save JSON artifacts<br/>outputs/artifacts/"]
    D3 --> N
    EC --> N
    EP --> N
    EN --> N
    E4 --> N
    F3 --> N
    G1 --> N
    H1 --> N
    I1 --> N
    J1 --> N
    K2 --> N
    K4 --> N
    L1 --> N
    M1 --> N

    N --> O["Runtime loading<br/>AppContext.load()"]
    O --> P["Similarity pipelines"]
```

## Short interpretation

```text
raresim/build/load_ontologies_to_local.py
    ↓
downloads raw source files to data/ontologies/

raresim/build/build_shared_artifacts.py
    ↓
builds reusable JSON artifacts in outputs/artifacts/

AppContext.load()
    ↓
loads artifacts for similarity methods
```
