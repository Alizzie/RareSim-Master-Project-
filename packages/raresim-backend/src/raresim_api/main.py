"""
FastAPI backend for RareSim web UI.

Endpoints:
    POST /api/extract   — extract HPO terms from raw text
    POST /api/diagnose  — run similarity methods and return ranked diseases

Run:
    uvicorn raresim_api.main:app --reload --port 8000
"""

import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from raresim.core.context import AppContext
from raresim.analysis.method_comparison import build_comparison
from raresim.core.pipeline import PipelineConfig
from raresim.hpo_extraction import build_patient_profile
from raresim.similarity_methods.autoencoder.pipeline import run as run_autoencoder
from raresim.similarity_methods.hpo2vec.pipeline import run as run_hpo2vec
from raresim.similarity_methods.llm.config import LLM_MODEL_LIST
from raresim.similarity_methods.llm.pipeline import run as run_llm
from raresim.similarity_methods.semantic.pipeline import run as run_semantic
from raresim.similarity_methods.set_based.pipeline import run as run_set_based
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf
from raresim.similarity_methods.transformer.config import MODEL_LIST as TRANSFORMER_MODEL_LIST
from raresim.similarity_methods.transformer.pipeline import run as run_transformer
from raresim.utils.hpo_utils import get_ancestors_inclusive, preprocess_ancestor_sets
from raresim.utils.io import load_json, save_json
from raresim.utils.paths import HPO_ANCESTORS_PATH, HPO_LABELS_PATH, WEBAPP_DIR
from raresim.utils.patient_loader import load_patient_with_extraction
from raresim.utils.paths import ARTIFACTS_DIR

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RareSim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load shared data once at startup ─────────────────────────────────────────
print("Loading HPO labels...")
hpo_labels = load_json(HPO_LABELS_PATH)
print(f"  {len(hpo_labels)} HPO labels loaded")

# ── Method groups ─────────────────────────────────────────────────────────────
SEMANTIC_METHODS = {
    "semantic_resnik_bma",
    "semantic_lin_bma",
    "semantic_jiang_conrath_bma",
}
SET_BASED_METHODS = {"set_cosine", "set_jaccard", "set_dice", "set_overlap", "set_jaccard_penalized"}
TFIDF_METHODS = {"tfidf_hpo", "tfidf_text", "tfidf_hybrid", "tfidf_hpo_labels"}
TRANSFORMER_MODEL_KEY_TO_METHOD: dict[str, str] = {
    "PubMedBERT": "transformer_PubMedBERT",
    "ClinicalBERT": "transformer_ClinicalBERT",
    "MiniLM": "transformer_MiniLM",
    "SapBERT": "transformer_SapBERT",
    "BioBERT": "transformer_BioBERT",
}


def _detect_transformer_model_key(model_name: str) -> str | None:
    """Return the canonical short key for one transformer model name."""
    normalized = model_name.lower().replace("-", "").replace("_", "")

    if "clinicalbert" in normalized:
        return "ClinicalBERT"
    if "pubmedbert" in normalized and "sapbert" not in normalized:
        return "PubMedBERT"
    if "minilm" in normalized:
        return "MiniLM"
    if "sapbert" in normalized:
        return "SapBERT"
    if "biobert" in normalized:
        return "BioBERT"
    return None


def _build_transformer_method_to_model() -> dict[str, str]:
    """Map accepted frontend method IDs to exact config.MODEL_LIST values."""
    method_to_model: dict[str, str] = {}

    for model_name in TRANSFORMER_MODEL_LIST:
        model_key = _detect_transformer_model_key(str(model_name))
        if model_key is None:
            continue

        method_key = TRANSFORMER_MODEL_KEY_TO_METHOD[model_key]
        method_to_model[method_key] = str(model_name)

    return method_to_model


TRANSFORMER_METHOD_TO_MODEL = _build_transformer_method_to_model()
TRANSFORMER_METHODS = set(TRANSFORMER_METHOD_TO_MODEL)
TRANSFORMER_GROUP_METHODS = {"transformer"}
LLM_METHODS = {"llm"}
HPO2VEC_METHODS = {"hpo2vec", "hpo2vec_plus"}
AUTOENCODER_METHODS = {"denoising_autoencoder"}

VALID_EXTRACTION_METHODS = {
    "dictionary",
    "biomedical_ner",
    "fast_hpo_cr",
    "chatgpt",
    "phenobrain_api",
}


# ── Request / Response models ─────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    """Request body for HPO extraction from clinical text."""

    text: str
    method: str = "dictionary"


class DiagnoseRequest(BaseModel):
    """Request body for running disease diagnosis/ranking."""

    mode: str
    hpo_terms: list[str] = Field(default_factory=list)
    excluded_hpo_terms: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    methods: list[str]
    top_k: int = 10


class SavePatientRequest(BaseModel):
    """Request body for saving a web patient profile and results."""

    patient_id: str
    raw_text: str = ""
    hpo_terms: list[str]
    results: list[dict[str, Any]]
    methods: list[str] = Field(default_factory=list)
    format: str = "json"


# ── Helper functions ──────────────────────────────────────────────────────────


def _all_supported_methods() -> set[str]:
    """Return all method keys accepted by the diagnose endpoint."""
    return (
        SEMANTIC_METHODS
        | SET_BASED_METHODS
        | TFIDF_METHODS
        | TRANSFORMER_METHODS
        | TRANSFORMER_GROUP_METHODS
        | LLM_METHODS
        | HPO2VEC_METHODS
        | AUTOENCODER_METHODS
    )


def _expand_selected_methods(selected: set[str]) -> set[str]:
    """Expand the frontend transformer button into transformer model methods."""
    expanded = set(selected)

    if "transformer" in expanded:
        expanded.remove("transformer")
        expanded.update(TRANSFORMER_METHODS)

    return expanded


def _validate_diagnose_request(req: DiagnoseRequest) -> None:
    """Validate one diagnosis request before expensive computation starts."""
    if not req.methods:
        raise HTTPException(status_code=400, detail="At least one method is required")

    unknown_methods = sorted(set(req.methods) - _all_supported_methods())
    if unknown_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method(s): {', '.join(unknown_methods)}",
        )

    if req.mode == "hpo" and not req.hpo_terms:
        raise HTTPException(status_code=400, detail="hpo_terms required for HPO mode")

    if req.mode == "text" and not req.raw_text:
        raise HTTPException(status_code=400, detail="raw_text required for text mode")


def _propagate_terms(hpo_terms: list[str]) -> list[str]:
    """Return sorted HPO terms plus all ontology ancestors."""
    ancestors = load_json(HPO_ANCESTORS_PATH)
    ancestor_sets = preprocess_ancestor_sets(ancestors)

    propagated: set[str] = set()
    for term in hpo_terms:
        propagated.update(get_ancestors_inclusive(term, ancestor_sets))

    return sorted(propagated)


def _build_patient(req: DiagnoseRequest):
    """Build a PatientProfile from frontend HPO terms and raw text."""
    hpo_terms = [term for term in req.hpo_terms if term not in req.excluded_hpo_terms]
    patient_dict = {
        "patient_id": "web_patient",
        "raw_text": req.raw_text or "",
        "hpo_terms": sorted(hpo_terms),
        "propagated_hpo_terms": _propagate_terms(hpo_terms),
        "excluded_hpo_terms": sorted(req.excluded_hpo_terms),
        "methods_used": ["web_input"],
    }

    tmp_path = WEBAPP_DIR / "web_patient_tmp.json"
    save_json(patient_dict, tmp_path)
    return load_patient_with_extraction(tmp_path, hpo_labels), hpo_terms


def _build_config(top_k: int) -> PipelineConfig:
    """Build the shared pipeline configuration used by the web endpoint."""
    return PipelineConfig(
        top_k=top_k,
        use_propagated_terms=True,
        ic_threshold=1.5,
        use_canonical_profiles=True,
    )


def _run_selected_methods(
    patient,
    selected: set[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, Any]:
    """Run all selected RareSim methods and return MethodResults by method."""
    all_results: dict[str, Any] = {}

    if selected & SEMANTIC_METHODS:
        methods = list(selected & SEMANTIC_METHODS)
        all_results.update(run_semantic(patient, methods, config, ctx))

    if selected & SET_BASED_METHODS:
        methods = list(selected & SET_BASED_METHODS)
        all_results.update(run_set_based(patient, methods, config, ctx))

    if selected & TFIDF_METHODS:
        all_results.update(run_tfidf(patient, list(selected & TFIDF_METHODS), config, ctx))

    if selected & TRANSFORMER_METHODS:
        method_keys = selected & TRANSFORMER_METHODS
        model_names = [
            TRANSFORMER_METHOD_TO_MODEL[key]
            for key in sorted(method_keys)
        ]
        try:
            transformer_results = run_transformer(patient, model_names, config, ctx)
        except Exception as error:
            raise RuntimeError(
                "Transformer run failed for "
                f"method_keys={sorted(method_keys)} and model_names={model_names}: "
                f"{error}"
            ) from error

        rekeyed_results = _rekey_and_relabel_transformers(transformer_results)
        all_results.update(rekeyed_results)

    if selected & LLM_METHODS:
        all_results.update(run_llm(patient, LLM_MODEL_LIST, config, ctx))

    if selected & HPO2VEC_METHODS:
        methods = list(selected & HPO2VEC_METHODS)
        all_results.update(run_hpo2vec(patient, methods, config, ctx))

    if selected & AUTOENCODER_METHODS:
        methods = list(selected & AUTOENCODER_METHODS)
        all_results.update(run_autoencoder(patient, methods, config, ctx))

    return all_results


def _iter_ranked_results(method_results: Any):
    """Yield ranked result rows from MethodResults or list-like objects."""
    ranked = (
        getattr(method_results, "rankings", None)
        or getattr(method_results, "ranked", None)
        or method_results
    )
    yield from ranked


def _transformer_result_key_to_method(raw_key: str) -> str:
    """Normalize transformer result keys into frontend method IDs."""
    if raw_key in TRANSFORMER_METHODS:
        return raw_key

    model_key = _detect_transformer_model_key(raw_key)
    if model_key is not None:
        return TRANSFORMER_MODEL_KEY_TO_METHOD[model_key]

    if raw_key.startswith("transformer_"):
        model_key = _detect_transformer_model_key(raw_key.removeprefix("transformer_"))
        if model_key is not None:
            return TRANSFORMER_MODEL_KEY_TO_METHOD[model_key]

    return raw_key


def _rekey_and_relabel_transformers(method_results: dict[str, Any]) -> dict[str, Any]:
    """
    Rename transformer MethodResults keys to frontend method IDs and relabel
    each SimilarityResult.method_name to match.
    """
    renamed: dict[str, Any] = {}

    for raw_key, results in method_results.items():
        clean_key = _transformer_result_key_to_method(str(raw_key))
        rows = list(_iter_ranked_results(results))

        for result in rows:
            result.method_name = clean_key

        renamed[clean_key] = results

    return renamed


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert one SimilarityResult-like object to the frontend shape."""
    return {
        "rank": result.rank,
        "disease_id": result.disease_id,
        "label": result.label,
        "score": result.score,
        "method_name": result.method_name,
        "shared_phenotype_labels": [],
        "explanation": getattr(result, "explanation", {}),
    }

ic_values = load_json(ARTIFACTS_DIR / "information_content.json")
MAX_IC = max(ic_values.values()) if ic_values else 1.0

def _flatten_results(all_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten MethodResults objects into one frontend result list."""
    flat_results: list[dict[str, Any]] = []

    for method_results in all_results.values():
        for result in _iter_ranked_results(method_results):
            flat_results.append(_result_to_dict(result))

    flat_results.sort(key=lambda row: row["score"], reverse=True)

    resnik_methods = {"semantic_resnik_bma"}
    method_max: dict[str, float] = {}
    for r in flat_results:
        if r["method_name"] in resnik_methods:
            method_max[r["method_name"]] = max(method_max.get(r["method_name"], 0.0), r["score"])
    for r in flat_results:
        if r["method_name"] in resnik_methods and method_max.get(r["method_name"], 0) > 0:
            r["score"] = r["score"] / method_max[r["method_name"]]
    return flat_results


def _limit_per_method(
    flat_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep only top-k rows per method and re-rank within each method."""
    per_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in flat_results:
        method = result["method_name"]
        if len(per_method[method]) < top_k:
            per_method[method].append(result)

    all_method_results = []
    for method_results in per_method.values():
        for rank, result in enumerate(method_results, start=1):
            result["rank"] = rank
        all_method_results.extend(method_results)

    return all_method_results


def _collect_by_method(all_results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build per-method ranked lists for the comparison component."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for method_results in all_results.values():
        for result in _iter_ranked_results(method_results):
            by_method[result.method_name].append(
                {
                    "disease_id": result.disease_id,
                    "label": result.label,
                    "score": result.score,
                    "rank": result.rank,
                }
            )

    return dict(by_method)


def _build_diagnose_response(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    all_results: dict[str, Any],
    hpo_terms: list[str],
    selected: set[str],
    ctx: AppContext,
    runtime_seconds: float,
    top_k: int,
) -> dict[str, Any]:
    """Build the JSON response for the diagnose endpoint."""
    flat_results = _flatten_results(all_results)
    all_method_results = _limit_per_method(flat_results, top_k)
    comparison = build_comparison(_collect_by_method(all_results), k=top_k, top_n=12)

    return {
        "results": all_method_results,
        "comparison": comparison,
        "meta": {
            "n_patient_terms": len(hpo_terms),
            "n_diseases": len(ctx.disease_profiles),
            "methods_run": sorted(selected),
            "runtime_seconds": round(runtime_seconds, 2),
        },
    }


def _build_phenopacket(req: SavePatientRequest, timestamp: str) -> dict[str, Any]:
    """Build a minimal phenopacket-style JSON object."""
    return {
        "id": req.patient_id,
        "subject": {"id": req.patient_id},
        "phenotypicFeatures": [
            {"type": {"id": term, "label": hpo_labels.get(term, term)}}
            for term in req.hpo_terms
        ],
        "metaData": {
            "created": datetime.now(timezone.utc).isoformat(),
            "resources": [
                {
                    "id": "hp",
                    "name": "Human Phenotype Ontology",
                    "namespacePrefix": "HP",
                }
            ],
            "raresim": {
                "methods_used": req.methods,
                "top_results": [
                    {
                        "disease_id": result.get("disease_id"),
                        "label": result.get("label"),
                        "score": result.get("score"),
                    }
                    for result in req.results[:5]
                ],
            },
        },
    }


def _build_patient_save_data(req: SavePatientRequest, timestamp: str) -> dict[str, Any]:
    """Build the JSON object saved for a normal web patient profile."""
    return {
        "patient_id": req.patient_id,
        "saved_at": timestamp,
        "raw_text": req.raw_text,
        "hpo_terms": req.hpo_terms,
        "methods": req.methods,
        "results": req.results,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/extract")
def extract(req: ExtractRequest):
    """Extract HPO terms from raw clinical text using the requested method."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    if req.method not in VALID_EXTRACTION_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    try:
        _, extracted = build_patient_profile(
            patient_id="web_extraction",
            raw_text=req.text,
            hpo_labels=hpo_labels,
            methods=[req.method],
        )
        return {"terms": extracted}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    """Run selected similarity methods and return ranked diseases."""
    try:
        _validate_diagnose_request(req)
        start = time.time()

        patient, hpo_terms = _build_patient(req)
        config = _build_config(req.top_k)
        ctx = AppContext.load(patient, config.use_canonical_profiles)
        requested = set(req.methods)
        selected = _expand_selected_methods(requested)

        all_results = _run_selected_methods(patient, selected, config, ctx)
        runtime_seconds = time.time() - start

        return _build_diagnose_response(
            all_results=all_results,
            hpo_terms=hpo_terms,
            selected=selected,
            ctx=ctx,
            runtime_seconds=runtime_seconds,
            top_k=req.top_k,
        )
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/health")
def health():
    """Return basic API health information."""
    return {"status": "ok", "hpo_labels_loaded": len(hpo_labels)}


@app.post("/api/patients/save")
def save_patient(req: SavePatientRequest):
    """Save a patient profile and diagnosis results from the web UI."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    methods_str = "_".join(req.methods) if req.methods else "unknown"
    folder = WEBAPP_DIR / "patient_profiles"
    folder.mkdir(parents=True, exist_ok=True)

    if req.format == "phenopacket":
        filename = f"{req.patient_id}_{methods_str}_{timestamp}.phenopacket.json"
        data = _build_phenopacket(req, timestamp)
    else:
        filename = f"{req.patient_id}_{methods_str}_{timestamp}.json"
        data = _build_patient_save_data(req, timestamp)

    save_json(data, folder / filename)
    return {"status": "saved", "filename": filename, "format": req.format}


@app.get("/api/hpo/search")
def hpo_search(q: str = ""):
    """Search loaded HPO labels by substring."""
    if not q.strip() or len(q) < 2:
        return {"terms": []}

    query = q.lower()
    results = [
        {"hpo_id": hpo_id, "label": label}
        for hpo_id, label in hpo_labels.items()
        if query in label.lower()
    ][:20]
    return {"terms": results}
