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
from raresim.core.method_comparison import build_comparison
from raresim.core.pipeline import PipelineConfig
from raresim.hpo_extraction import build_patient_profile
from raresim.similarity_methods.autoencoder.pipeline import run as run_autoencoder
from raresim.similarity_methods.hpo2vec.pipeline import run as run_hpo2vec
from raresim.similarity_methods.llm.config import LLM_MODEL_LIST
from raresim.similarity_methods.llm.pipeline import run as run_llm
from raresim.similarity_methods.semantic.pipeline import run as run_semantic
from raresim.similarity_methods.set_based.pipeline import run as run_set_based
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf
from raresim.similarity_methods.transformer.pipeline import (
    run_default_model as run_transformer,
)
from raresim.utils.hpo_utils import get_ancestors_inclusive, preprocess_ancestor_sets
from raresim.utils.io import load_json, save_json
from raresim.utils.paths import HPO_ANCESTORS_PATH, HPO_LABELS_PATH, WEBAPP_DIR
from raresim.utils.patient_loader import load_patient_with_extraction

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
SET_BASED_METHODS = {"set_cosine", "set_jaccard", "set_dice", "set_overlap"}
TFIDF_METHODS = {"tfidf_hpo", "tfidf_text", "tfidf_hybrid", "tfidf_hpo_labels"}
TRANSFORMER_METHODS = {"transformer"}
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


def _validate_diagnose_request(req: DiagnoseRequest) -> None:
    """Validate one diagnosis request before expensive computation starts."""
    if not req.methods:
        raise HTTPException(status_code=400, detail="At least one method is required")

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
        all_results.update(run_transformer(patient, config, ctx))

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


def _flatten_results(all_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten MethodResults objects into one frontend result list."""
    flat_results: list[dict[str, Any]] = []

    for method_results in all_results.values():
        for result in _iter_ranked_results(method_results):
            flat_results.append(_result_to_dict(result))

    flat_results.sort(key=lambda row: row["score"], reverse=True)
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
            "methods_run": list(selected),
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
        selected = set(req.methods)

        print(f"DEBUG: disease profiles loaded: {len(ctx.disease_profiles)}")

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
