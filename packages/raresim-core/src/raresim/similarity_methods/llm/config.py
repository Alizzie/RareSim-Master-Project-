"""
Configuration for the LLM-based disease retrieval and explanation pipeline.

Backend: HuggingFace transformers (GPU server required)

Models (generative/decoder models):
- Mistral/Mistral-7B-Instruct-v0.2 : works ok at the moment
"""

from raresim.utils.paths import TRANSFORMER_DIR, SIMILARITY_DIR

PIPELINE_NAME = "llm"
LLM_DIR = SIMILARITY_DIR / PIPELINE_NAME
LLM_DIR.mkdir(parents=True, exist_ok=True)

# Transformer results to use as input for explainer
TRANSFORMER_RESULTS_PATH = TRANSFORMER_DIR / "all_model_results_summary_canonical.json"

# Models to run in the LLM pipeline
LLM_MODEL_LIST = [
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# Model used for explaining transformer/direct LLM results
EXPLAINER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ── Generation settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS_RETRIEVAL = 2048
MAX_NEW_TOKENS_EXPLAINER = 1024
TEMPERATURE = 0.1
DO_SAMPLE = True

REPETITION_PENALTY = 1.3
TEXT_PREVIEW_MAX_LENGTH = 400
DISEASE_HPO_TERMS_PREVIEW_MAX_COUNT = 20

# ── Pipeline settings ─────────────────────────────────────────────────────────
TOP_K = 10
TOP_K_RERANK = 10

# ── Match-level scoring ───────────────────────────────────────────────────────

MATCH_LEVEL_SCORES: dict[str, float] = {
    "strong": 0.9,
    "possible": 0.6,
    "weak": 0.3,
    "unlikely": 0.1,
}

# Used when the LLM text can't be mapped to a known level.
DEFAULT_MATCH_SCORE = 0.5

# Raw LLM vocabulary -> canonical level.
MATCH_LEVEL_ALIASES: dict[str, str] = {
    "high": "strong",
    "strong": "strong",
    "medium": "possible",
    "moderate": "possible",
    "possible": "possible",
    "low": "weak",
    "weak": "weak",
    "unlikely": "unlikely",
}
