"""
Central registry of similarity methods.

Single source of truth for:
  - METHOD_MODULES : pipeline_name → module, for uniform dispatch. Each module
                     exposes run(patient, selected, config, ctx) and ALL_METHODS.
  - the flat per-group method lists (SEMANTIC_METHODS, etc.), derived from each
    package's own config — never hardcoded here.
  - ALL_METHODS : every method name across all packages.
  - DEFAULTS : default CLI configuration.

Adding a method package requires only:
  1. adding it to METHOD_MODULES below, and
  2. ensuring its package exposes run() and ALL_METHODS.
Everything else (flat lists, ALL_METHODS, CLI choices) updates automatically.
"""

from raresim.utils.paths import EXAMPLE_PATIENT_PATH

# ── Method packages ───────────────────────────────────────────────────────────
# Each package's __init__ exposes: run(patient, selected, config, ctx) and ALL_METHODS.
from raresim.similarity_methods import (
    semantic,
    set_based,
    tfidf,
    hpo2vec,
    autoencoder,
    transformer,
    llm,
)

# ── Dispatch registry ─────────────────────────────────────────────────────────
# pipeline_name → module. app.py loops over this to run selected methods.
METHOD_MODULES = {
    "semantic": semantic,
    "set_based": set_based,
    "tfidf": tfidf,
    "hpo2vec": hpo2vec,
    "autoencoder": autoencoder,
    "transformer": transformer,
    "llm": llm,
}

# ── Per-group method lists (derived from each package) ────────────────────────
SEMANTIC_METHODS = list(semantic.METHOD_NAMES)
SET_BASED_METHODS = list(set_based.METHOD_NAMES)
TFIDF_METHODS = list(tfidf.METHOD_NAMES)
HPO2VEC_METHODS = list(hpo2vec.METHOD_NAMES)
AUTOENCODER_METHODS = list(autoencoder.METHOD_NAMES)
TRANSFORMER_METHODS = list(transformer.METHOD_NAMES)
LLM_METHODS = list(llm.METHOD_NAMES)

# ── Aggregate ─────────────────────────────────────────────────────────────────
ALL_METHODS: list[str] = [
    method for module in METHOD_MODULES.values() for method in module.METHOD_NAMES
]

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    "patient_path": EXAMPLE_PATIENT_PATH,
    "methods": ALL_METHODS,
    "top_k": 10,
    "use_propagated_terms": True,
    "ic_threshold": 1.5,
    "use_canonical_profiles": True,
}

__all__ = [
    "METHOD_MODULES",
    "ALL_METHODS",
    "SEMANTIC_METHODS",
    "SET_BASED_METHODS",
    "TFIDF_METHODS",
    "HPO2VEC_METHODS",
    "AUTOENCODER_METHODS",
    "TRANSFORMER_METHODS",
    "LLM_METHODS",
    "DEFAULTS",
]
