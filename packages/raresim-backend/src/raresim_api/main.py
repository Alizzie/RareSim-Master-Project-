"""
FastAPI backend for RareSim web UI.

Endpoints:
    POST /api/extract   — extract HPO terms from raw text
    POST /api/diagnose  — run similarity methods and return ranked diseases

Run:
    uvicorn raresim_api.main:app --reload --port 8000
"""

import traceback
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── RareSim imports ───────────────────────────────────────────────────────────
from raresim.core.context import AppContext
from raresim.utils.io import load_json, save_json
from raresim.utils.patient_loader import load_patient_with_extraction
from raresim.utils.paths import (
    WEBAPP_DIR,
    ALIAS_TO_CANONICAL_PATH,
    HPO_LABELS_PATH,
    HPO_ANCESTORS_PATH,
)
from raresim.utils.hpo_utils import (
    preprocess_ancestor_sets,
    get_ancestors_inclusive,
)
from raresim.core.pipeline import PipelineConfig
from raresim.hpo_extraction import build_patient_profile

from raresim.similarity_methods.semantic.pipeline import run as run_semantic
from raresim.similarity_methods.set_based.pipeline import run as run_set_based
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf
from raresim.similarity_methods.transformer.pipeline import run as run_transformer
from raresim.similarity_methods.llm.pipeline import run as run_llm
from raresim.similarity_methods.hpo2vec.pipeline import run as run_hpo2vec
from raresim.similarity_methods.autoencoder.pipeline import run as run_autoencoder

from raresim.core.method_comparison import build_comparison

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
TFIDF_METHODS = {"tfidf"}
TRANSFORMER_METHODS = {"transformer"}
LLM_METHODS = {"llm"}
HPO2VEC_METHODS = {"hpo2vec_plus"}
AUTOENCODER_METHODS = {"denoising_autoencoder"}

from collections import defaultdict

def _collect_by_method(all_results, all_raw_results):
    """Per-method ranked lists, keeping each method's own ranks (for comparison)."""
    by_method = defaultdict(list)
    for method_results in all_results.values():
        ranked = (getattr(method_results, "rankings", None)
                  or getattr(method_results, "ranked", None) or method_results)
        for r in ranked:
            by_method[r.method_name].append(
                {"disease_id": r.disease_id, "label": r.label, "score": r.score, "rank": r.rank})
    for method_name, results_list in all_raw_results.items():
        if isinstance(results_list, dict):           # transformer: {model: [...]}
            for model_name, model_results in results_list.items():
                name = f"{method_name}_{model_name}"
                for r in model_results:
                    by_method[name].append(
                        {"disease_id": r.get("canonical_disease_id") or r.get("disease_id", ""),
                         "label": r.get("label", ""), "score": r.get("score", 0.0),
                         "rank": r.get("rank", 0)})
        elif isinstance(results_list, list):         # llm
            for r in results_list:
                by_method[method_name].append(
                    {"disease_id": r.get("disease_id", ""), "label": r.get("label", ""),
                     "score": r.get("score", 0.0), "rank": r.get("rank", 0)})
    return dict(by_method)

# ── Request / Response models ─────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    text: str
    method: str = "dictionary"


class DiagnoseRequest(BaseModel):
    mode: str
    hpo_terms: list[str] = []
    excluded_hpo_terms: list[str] = []
    raw_text: Optional[str] = None
    methods: list[str]
    top_k: int = 10

class SavePatientRequest(BaseModel):
    patient_id: str
    raw_text: str = ""
    hpo_terms: list[str]
    results: list[dict]
    methods: list[str] = []
    format: str = "json"  # "json" or "phenopacket"

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/extract")
def extract(req: ExtractRequest):
    """
    Extract HPO terms from raw clinical text using the requested method.
    Returns a list of { hpo_id, label, method, confidence } dicts.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    valid_methods = {
        "dictionary",
        "biomedical_ner",
        "fast_hpo_cr",
        "chatgpt",
        "phenobrain_api",
    }
    if req.method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    try:
        _, extracted = build_patient_profile(
            patient_id="web_extraction",
            raw_text=req.text,
            hpo_labels=hpo_labels,
            methods=[req.method],
        )
        return {"terms": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    """
    Run similarity diagnosis and return ranked diseases.

    Input modes:
        mode='hpo'  — use req.hpo_terms directly
        mode='text' — use req.raw_text for transformer/llm,
                      req.hpo_terms for semantic/set-based (pre-extracted)
    """

    try:
        if not req.methods:
            raise HTTPException(
                status_code=400, detail="At least one method is required"
            )

        if req.mode == "hpo" and not req.hpo_terms:
            raise HTTPException(
                status_code=400, detail="hpo_terms required for HPO mode"
            )

        if req.mode == "text" and not req.raw_text:
            raise HTTPException(
                status_code=400, detail="raw_text required for text mode"
            )

        start = time.time()

        # ── Build patient dict ────────────────────────────────────────────────────

        ancestors = load_json(HPO_ANCESTORS_PATH)
        ancestor_sets = preprocess_ancestor_sets(ancestors)

        hpo_terms = req.hpo_terms
        hpo_terms = [t for t in req.hpo_terms if t not in req.excluded_hpo_terms]
        propagated: set = set()
        for term in hpo_terms:
            propagated |= get_ancestors_inclusive(term, ancestor_sets)

        patient_dict = {
            "patient_id": "web_patient",
            "raw_text": req.raw_text or "",
            "hpo_terms": sorted(hpo_terms),
            "propagated_hpo_terms": sorted(propagated),
            "methods_used": ["web_input"],
        }

        tmp_path = WEBAPP_DIR / "web_patient_tmp.json"
        save_json(patient_dict, tmp_path)
        patient = load_patient_with_extraction(tmp_path, hpo_labels)

        # ── Config ────────────────────────────────────────────────────────────────
        config = PipelineConfig(
            top_k=req.top_k,
            use_propagated_terms=True,
            ic_threshold=1.5,
            use_canonical_profiles=True,
        )

        # ── AppContext ────────────────────────────────────────────────────────────
        ctx = AppContext.load(patient, config.use_canonical_profiles)
        print(f"DEBUG: disease profiles loaded: {len(ctx.disease_profiles)}")

        # ── Run selected methods ──────────────────────────────────────────────────
        all_results = {}
        all_raw_results = {}
        selected = set(req.methods)

        if selected & SEMANTIC_METHODS:
            all_results.update(run_semantic(patient, list(selected), config, ctx))

        if selected & SET_BASED_METHODS:
            all_results.update(run_set_based(patient, list(selected), config, ctx))

        if selected & TFIDF_METHODS:
            all_results.update(run_tfidf(patient, list(selected), config, ctx))

        if selected & TRANSFORMER_METHODS:
            alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
            raw = run_transformer(
                disease_profiles=ctx.disease_profiles,
                hpo_labels=hpo_labels,
                patient=patient_dict,
                alias_to_canonical=alias_to_canonical,
                top_k=config.top_k,
            )
            all_raw_results.update(raw)

        if selected & LLM_METHODS:
            all_raw_results["llm"] = run_llm(
                patient=patient_dict,
                hpo_labels=hpo_labels,
                disease_profiles=ctx.disease_profiles,
                top_k=config.top_k,
            )

        if selected & HPO2VEC_METHODS:
            all_results.update(run_hpo2vec(patient, list(selected), config, ctx))

        if selected & AUTOENCODER_METHODS:
            all_results.update(run_autoencoder(patient, list(selected), config, ctx))

        # ── Flatten to unified result list ────────────────────────────────────────
        flat_results = []

        # SimilarityResult objects → dicts
        for method_results in all_results.values():
            # MethodResults has a .rankings attribute, not .ranked
            ranked = (
                getattr(method_results, "rankings", None)
                or getattr(method_results, "ranked", None)
                or method_results
            )
            for r in ranked:
                flat_results.append(
                    {
                        "rank": r.rank,
                        "disease_id": r.disease_id,
                        "label": r.label,
                        "score": r.score,
                        "method_name": r.method_name,
                        "shared_phenotype_labels": [],
                        "explanation": (
                            r.explanation if hasattr(r, "explanation") else {}
                        ),
                    }
                )

        # Raw results (transformer, llm) → same shape
        for method_name, results_list in all_raw_results.items():
            if isinstance(results_list, dict):
                # transformer returns {model_name: [...]}
                for model_name, model_results in results_list.items():
                    for r in model_results:
                        flat_results.append(
                            {
                                "rank": r.get("rank", 0),
                                "disease_id": r.get("canonical_disease_id")
                                or r.get("disease_id", ""),
                                "label": r.get("label", ""),
                                "score": r.get("score", 0.0),
                                "method_name": f"{method_name}_{model_name}",
                                "shared_phenotype_labels": r.get("explanation", {}).get(
                                    "shared_phenotype_labels", []
                                ),
                                "explanation": r.get("explanation", {}),
                            }
                        )
            elif isinstance(results_list, list):
                for r in results_list:
                    flat_results.append(
                        {
                            "rank": r.get("rank", 0),
                            "disease_id": r.get("disease_id", ""),
                            "label": r.get("label", ""),
                            "score": r.get("score", 0.0),
                            "method_name": method_name,
                            "shared_phenotype_labels": [],
                            "explanation": {},
                        }
                    )

        # Sort by score descending, re-rank
        flat_results.sort(key=lambda x: x["score"], reverse=True)

        per_method = {}
        for r in flat_results:
            method = r["method_name"]
            if method not in per_method:
                per_method[method] = []
            if len(per_method[method]) < req.top_k:
                per_method[method].append(r)

        all_method_results = []
        for method_results_list in per_method.values():
            for rank, r in enumerate(method_results_list, start=1):
                r["rank"] = rank
            all_method_results.extend(method_results_list)

        by_method = _collect_by_method(all_results, all_raw_results)
        comparison = build_comparison(by_method, k=req.top_k, top_n=12)

        return {
            "results": all_method_results,
            "comparison": comparison,
            "meta": {
                "n_patient_terms": len(hpo_terms),
                "n_diseases": len(ctx.disease_profiles),
                "methods_run": list(selected),
                "runtime_seconds": round(time.time() - start, 2),
            },
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "hpo_labels_loaded": len(hpo_labels)}

@app.post("/api/patients/save")
def save_patient(req: SavePatientRequest):
    from datetime import datetime, timezone

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    methods_str = "_".join(req.methods) if req.methods else "unknown"
    folder = WEBAPP_DIR / "patient_profiles"
    folder.mkdir(parents=True, exist_ok=True)

    if req.format == "phenopacket":
        filename = f"{req.patient_id}_{methods_str}_{timestamp}.phenopacket.json"
        data = {
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
                        {"disease_id": r.get("disease_id"), "label": r.get("label"), "score": r.get("score")}
                        for r in req.results[:5]
                    ],
                },
            },
        }
    else:
        filename = f"{req.patient_id}_{methods_str}_{timestamp}.json"
        data = {
            "patient_id": req.patient_id,
            "saved_at": timestamp,
            "raw_text": req.raw_text,
            "hpo_terms": req.hpo_terms,
            "methods": req.methods,
            "results": req.results,
        }

    save_json(data, folder / filename)
    return {"status": "saved", "filename": filename, "format": req.format}

@app.get("/api/hpo/search")
def hpo_search(q: str = ""):
    if not q.strip() or len(q) < 2:
        return {"terms": []}
    q_lower = q.lower()
    results = [
        {"hpo_id": hpo_id, "label": label}
        for hpo_id, label in hpo_labels.items()
        if q_lower in label.lower()
    ][:20]
    return {"terms": results}
