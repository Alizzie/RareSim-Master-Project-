"""
raresim.ontology — building and querying ontology-derived data.

This package turns raw ontology sources (HPO, ORDO, MONDO, HOOM, HPOA,
Monarch) into the artifacts that pipelines consume: disease profiles,
information content values, ancestor maps, and disease categories.

Dependency layering
--------------------
ontology/ depends on: types/, utils/  (and third-party ontology libs)
ontology/ must NOT depend on: core/, similarity_methods/

Distinction from utils/hpo_utils.py
------------------------------------
ontology/hpo_utils.py  — BUILDS ontology structures from raw files
                         (compute_ancestors, propagate_hpo_terms). Run once
                         during artifact construction.
utils/hpo_utils.py     — USES pre-built structures at pipeline runtime
                         (filter_terms_by_ic, preprocess_ancestor_sets).
"""

# ── Disease profile construction ──────────────────────────────────────────────
# from raresim.ontology.disease_profiles import (
#     build_canonical_disease_profiles,
#     expand_alias_profiles,
# )

# ── HPO ontology building (ancestors, propagation) ────────────────────────────
# from raresim.ontology.hpo_utils import (
#     compute_ancestors,
#     propagate_hpo_terms,
# )

# ── Information content ────────────────────────────────────────────────────────
# from raresim.ontology.ic import (
#     compute_information_content,
# )

# ── Disease ancestor / category structure ─────────────────────────────────────
# from raresim.ontology.disease_ancestors import (
#     build_disease_ancestors,
# )
from raresim.ontology.disease_category import (
    build_category_metadata,
)

# ── Ontology source loaders ───────────────────────────────────────────────────
# from raresim.ontology.loaders import (
#     load_ordo_metadata,
#     load_mondo_metadata,
#     load_hoom_metadata,
#     load_hpo_labels,
#     load_phenotype_annotations,
# )

# ── Phenotype merging ─────────────────────────────────────────────────────────
# from raresim.ontology.phenotype_merge import (
#     merge_phenotype_annotations,
# )


__all__ = [
    # ── disease profiles ──
    # "build_canonical_disease_profiles",
    # "expand_alias_profiles",
    # ── hpo building ──
    # "compute_ancestors",
    # "propagate_hpo_terms",
    # ── information content ──
    # "compute_information_content",
    # ── disease ancestors / categories ──
    # "build_disease_ancestors",
    "build_category_metadata",
    # ── loaders ──
    # "load_ordo_metadata",
    # "load_mondo_metadata",
    # "load_hoom_metadata",
    # "load_hpo_labels",
    # "load_phenotype_annotations",
    # ── phenotype merge ──
    # "merge_phenotype_annotations",
]
