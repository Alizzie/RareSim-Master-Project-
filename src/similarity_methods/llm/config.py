"""
Configuration for the LLM-based disease retrieval and explanation pipeline.

Backends:
- "ollama"   : local Ollama server
               Install: brew install ollama
               Pull:    ollama pull mistral
               Start:   ollama serve
- "hf"       : HuggingFace transformers (heavy, use on GPU server)
               BioMistral/BioMistral-7B is the best biomedical option

Input paths come from shared.paths — not defined here.
"""

from shared.paths import PROJECT_ROOT, TRANSFORMER_DIR

LLM_DIR = PROJECT_ROOT / "outputs" / "llm"
LLM_DIR.mkdir(parents=True, exist_ok=True)

# Transformer results to use as input for explainer
TRANSFORMER_RESULTS_PATH = (
    TRANSFORMER_DIR / "all_model_results_summary_canonical.json"
)

# ── Backend switch ────────────────────────────────────────────────────────────
LLM_BACKEND = "ollama"

# ── Ollama settings ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_RETRIEVAL_MODEL = "mistral"
OLLAMA_EXPLAINER_MODEL = "mistral"

# ── HuggingFace settings (GPU server) ────────────────────────────────────────
HF_RETRIEVAL_MODEL = "BioMistral/BioMistral-7B"
HF_EXPLAINER_MODEL = "BioMistral/BioMistral-7B"

# ── Active models (set automatically based on backend) ───────────────────────
RETRIEVAL_MODEL = OLLAMA_RETRIEVAL_MODEL if LLM_BACKEND == "ollama" else HF_RETRIEVAL_MODEL
EXPLAINER_MODEL = OLLAMA_EXPLAINER_MODEL if LLM_BACKEND == "ollama" else HF_EXPLAINER_MODEL

# ── Generation settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS_RETRIEVAL = 1024
MAX_NEW_TOKENS_EXPLAINER = 1024
TEMPERATURE = 0.1
DO_SAMPLE = False

# ── Pipeline settings ─────────────────────────────────────────────────────────
TOP_K = 10
TOP_K_RERANK = 10
