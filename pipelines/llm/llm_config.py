from pathlib import Path

"""
Configuration for the LLM-based disease retrieval and explanation pipeline.

Backends:
- "ollama"   : local Ollama server
               Install: brew install ollama
               Pull:    ollama pull mistral
               Start:   ollama serve
- "hf"       : HuggingFace transformers (heavy, we can use on GPU server)
               BioMistral/BioMistral-7B is the best biomedical option

Switch backend with LLM_BACKEND below.
On GPU server → switch to "hf" + BioMistral for better biomedical quality.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SHARED_DIR = OUTPUTS_DIR / "shared"
LLM_DIR = OUTPUTS_DIR / "llm"
TRANSFORMER_DIR = OUTPUTS_DIR / "transformer"

LLM_DIR.mkdir(parents=True, exist_ok=True)

# Input paths
DISEASE_PROFILES_PATH = SHARED_DIR / "disease_profiles.json"
HPO_LABELS_PATH = SHARED_DIR / "hpo_labels.json"
PATIENT_PATH = SHARED_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = SHARED_DIR / "alias_to_canonical.json"

# Transformer results to use as input for explainer
TRANSFORMER_RESULTS_PATH = (
    TRANSFORMER_DIR / "all_model_results_summary_canonical.json"
)

# ── Backend switch ────────────────────────────────────────────────────────────
# "ollama" → local Ollama server, no loading delay, good for development
# "hf"     → HuggingFace transformers, heavy but best biomedical quality
LLM_BACKEND = "ollama"

# ── Ollama settings ───────────────────────────────────────────────────────────
# Make sure ollama is running: ollama serve
# Available models: mistral, llama3
# Pull a model first: ollama pull mistral
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_RETRIEVAL_MODEL = "mistral"     
OLLAMA_EXPLAINER_MODEL = "mistral"     # or "llama3" for better reasoning

# ── HuggingFace settings (GPU server) ────────────────────────────────────────
# biomedical model, use on uni GPU server
HF_RETRIEVAL_MODEL = "BioMistral/BioMistral-7B"
HF_EXPLAINER_MODEL = "BioMistral/BioMistral-7B"

# ── Active models (set automatically based on backend) ───────────────────────
RETRIEVAL_MODEL = OLLAMA_RETRIEVAL_MODEL if LLM_BACKEND == "ollama" else HF_RETRIEVAL_MODEL
EXPLAINER_MODEL = OLLAMA_EXPLAINER_MODEL if LLM_BACKEND == "ollama" else HF_EXPLAINER_MODEL

# ── Generation settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS_RETRIEVAL = 1024
MAX_NEW_TOKENS_EXPLAINER = 1024
TEMPERATURE = 0.1       # low = more deterministic, good for medical tasks
DO_SAMPLE = False       # greedy decoding for reproducibility

# ── Pipeline settings ─────────────────────────────────────────────────────────
TOP_K = 10              # top-K diseases to retrieve
TOP_K_RERANK = 10       # top-K transformer results to explain
