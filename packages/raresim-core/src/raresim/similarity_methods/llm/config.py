"""
Configuration for the LLM-based disease retrieval and explanation pipeline.

Backend: HuggingFace transformers (GPU server required)

Models (generative/decoder models):
- Mistral/Mistral-7B-Instruct-v0.2 : works ok at the moment
"""

from raresim.utils.paths import LLM_DIR, TRANSFORMER_DIR

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

# ── Pipeline settings ─────────────────────────────────────────────────────────
TOP_K = 10
TOP_K_RERANK = 10
