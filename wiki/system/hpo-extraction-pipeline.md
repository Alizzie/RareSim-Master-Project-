# HPO Extraction Pipeline

## Purpose

The `hpo_extraction` module converts raw patient text into structured phenotype terms. It's kept separate from the similarity methods because extraction is a preprocessing task, not a disease-ranking method — its output becomes part of the `PatientProfile`, usable by any similarity method afterward.

For the confirmed function signatures, extraction method keys, and `skip_negated` behavior, see [Patient Profile Construction](/artifacts/patient-profile-construction) in Artifacts — this page covers the pipeline shape end-to-end at the architecture level.

## Pipeline Flow

```mermaid
flowchart TD
    TEXT["Raw clinical text"] --> BUILD["build_patient_profile()<br/>hpo_extraction/ensemble.py"]
    BUILD --> DISPATCH["extract_hpo_terms()<br/>dispatch to selected extractors"]

    DISPATCH --> D1["dictionary.py<br/>exact label matching"]
    DISPATCH --> D2["ner.py<br/>biomedical NER"]
    DISPATCH --> D3["fast_hpo_cr.py<br/>FastHPOCR"]
    DISPATCH --> D4["gpt.py<br/>GPT-based extraction"]
    DISPATCH --> D5["phenobrain.py<br/>PhenoBrain API"]

    D1 --> EXTRES["ExtractionResult objects<br/>hpo_id, label, matched_text,<br/>method, confidence, negation"]
    D2 --> EXTRES
    D3 --> EXTRES
    D4 --> EXTRES
    D5 --> EXTRES

    EXTRES --> DEDUP["deduplicate()<br/>drop blocklisted terms,<br/>keep highest confidence per HPO ID"]
    DEDUP --> SPLIT["Separate by negation flag"]

    SPLIT --> DIRECT["Positive terms<br/>sorted unique HPO IDs"]
    SPLIT --> NEGTERM["Excluded terms<br/>negation flag = true"]

    DIRECT --> EXPAND["Expand with hpo_ancestors.json"]
    EXPAND --> PROP["propagated_hpo_terms"]

    PROP --> DICT["Patient dict<br/>hpo_terms, propagated_hpo_terms,<br/>excluded_hpo_terms, methods_used"]
    DIRECT --> DICT
    NEGTERM --> DICT

    DICT --> PROFILE["PatientProfile<br/>utils/patient_loader.py"]
```
