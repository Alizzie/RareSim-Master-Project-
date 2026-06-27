"""
Build-time behaviour constants for RareSim.

Controls how artifacts are built — flags, thresholds, and seed data.
No path constants belong here; those live in utils/paths.py.

Import pattern:
    from raresim.core.config import APPLY_TRUE_PATH_RULE, EXAMPLE_PATIENT
"""

# ── Disease profile build settings ────────────────────────────────────────────

# Apply the true-path rule during HPO term propagation.
# When True, all ancestor terms of each annotated HPO term are added
# to the disease profile (standard practice for HPO-based similarity).
APPLY_TRUE_PATH_RULE = True

# Minimum number of HPO terms a disease profile must have to be included.
# Profiles below this threshold are filtered out as too sparse for matching.
MIN_DISEASE_HPO_TERMS = 1

# Canonical disease namespace — profiles are preferentially keyed by
# Orphanet IDs when a mapping exists.
CANONICAL_DISEASE_NAMESPACE = "ORPHA"

# ── Example patient ───────────────────────────────────────────────────────────

# Used by build_shared_artifacts.py to generate an example patient profile
# and by pipeline scripts as a default test case.
EXAMPLE_PATIENT = {
    "patient_id": "patient_001",
    "raw_text": "Patient with developmental delay, cerebellar ataxia, and anemia.",
    "hpo_terms": ["HP:0001263", "HP:0002470", "HP:0001903"],
}
