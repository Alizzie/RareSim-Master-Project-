"""
Phenotype extraction pipeline.

Extracts HPO terms from raw clinical text using these methods:
1. Dictionary      — exact HPO label matching (fast baseline)
2. Biomedical NER  — d4data transformer NER + HPO label lookup
3. FastHPOCR       — morphological token cluster dictionary matching
                     Clone: git clone https://github.com/tudorgroza/fast_hpo_cr.git src/fast_hpo_cr
                     Download hp.obo since it uses this format for hpo terms: wget https://purl.obolibrary.org/obo/hp.obo -O ontologies/model/hp.obo
4. ChatGPT         — GPT-4o-mini prompted to extract HPO IDs from clinical text - can hallucinate
                     Paper: https://ieeexplore.ieee.org/document/10340611
                     Requires: pip install openai
                     Add key to env file: OPENAI_API_KEY=sk-...
5. PhenoBrain API  — BERT NER via PhenoBrain public web API (no key needed)
                     Endpoint: https://www.phenobrain.cs.tsinghua.edu.cn/extract-hpo

Usage:
    from shared.phenotype import build_patient_profile, extract_hpo_terms
    from shared.io import load_json
    from shared.paths import HPO_LABELS_PATH
    hpo_labels = load_json(HPO_LABELS_PATH)

    patient, extracted_terms = build_patient_profile(
        patient_id="patient_001",
        raw_text="Patient with cerebellar ataxia and anemia.",
        hpo_labels=hpo_labels,
        methods=["dictionary", "biomedical_ner",
                 "fast_hpo_cr", "chatgpt", "phenobrain_api"],
    )

    print(f"Extracted {len(extracted_terms)} HPO terms:")
    for t in extracted_terms:
        print(f"  {t['hpo_id']} | {t['label']} | method={t['method']}")
"""

import json
import os
import re
import sys
import time
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.config import (
    BIOMEDICAL_NER_MIN_CONFIDENCE,
    BIOMEDICAL_NER_MODEL,
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
    EXTRACTION_METHODS,
)
from shared.io import load_json, save_json
from shared.paths import (
    HPO_LABELS_PATH,
    OUTPUTS_DIR,
    OUTPUT_EXTRACTION_PATH,
    OUTPUT_PATIENT_PATH,
    PATIENT_PATH,
    PHENOTYPE_DIR,
    PROJECT_ROOT,
    HPO_ANCESTORS_PATH
)
from shared.math import preprocess_ancestor_sets, get_ancestors_inclusive

# ── FastHPOCR paths ────────────────────────────────────────────────────────────
_FAST_HPO_CR_SRC     = PROJECT_ROOT / "src" / "fast_hpo_cr"
_HP_OBO_PATH         = PROJECT_ROOT / "ontologies" / "model" / "hp.obo"
_FAST_HPO_CR_IDX_DIR = OUTPUTS_DIR / "fast_hpo_cr_index"

# ── PhenoBrain API ─────────────────────────────────────────────────────────────
_PHENOBRAIN_SUBMIT_URL = "https://www.phenobrain.cs.tsinghua.edu.cn/extract-hpo"
_PHENOBRAIN_RESULT_URL = "https://www.phenobrain.cs.tsinghua.edu.cn/query-extract-hpo-result"
_PHENOBRAIN_POLL_INTERVAL = 2    # seconds between polls
_PHENOBRAIN_MAX_POLLS     = 30   # give up after 60s

# ── ChatGPT settings ───────────────────────────────────────────────────────────
_CHATGPT_MODEL = "gpt-4o-mini"
_CHATGPT_SYSTEM_PROMPT = """You are a clinical phenotype extraction expert.

Given clinical text, extract only explicitly mentioned abnormal human phenotype phrases.

Rules:
- Return ONLY valid JSON.
- Do not return markdown.
- Do not return explanations.
- Do not infer unstated phenotypes.
- Do not return HPO IDs.
- Do not include diagnoses, disease names, genes, treatments, inheritance patterns, or normal findings.
- Extract at most 20 phenotype phrases.
- Use short canonical medical phrases such as "microcephaly", "global developmental delay", or "hypotonia".

Output format:
{"phenotypes": ["microcephaly", "global developmental delay", "hypotonia"]}

If no phenotype is found, return:
{"phenotypes": []}
"""


# ── Data types ────────────────────────────────────────────────────────────────


class ExtractionMethod(str, Enum):
    DICTIONARY     = "hpo_label_dictionary_match"
    BIOMEDICAL_NER = "biomedical_ner_d4data"
    FAST_HPO_CR    = "fast_hpo_cr"
    CHATGPT        = "chatgpt_extraction"
    PHENOBRAIN_API = "phenobrain_api"


@dataclass
class ExtractionResult:
    hpo_id: str
    label: str
    matched_text: str
    method: ExtractionMethod
    confidence: float = 1.0
    start: Optional[int] = None
    end: Optional[int] = None
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "hpo_id": self.hpo_id,
            "label": self.label,
            "matched_text": self.matched_text,
            "method": self.method.value,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
            "negated": self.negated,
        }


# ── Shared utils ──────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_negated(
    text: str,
    start_index: int,
    window_size: int = NEGATION_WINDOW_SIZE,
) -> bool:
    """Check whether a phenotype mention is negated."""
    before = text[max(0, start_index - window_size): start_index]
    return any(neg in before for neg in NEGATION_WORDS)


def build_label_lookup(hpo_labels: Dict[str, str]) -> Dict[str, str]:
    """Build a normalized label to HPO ID lookup."""
    return {
        normalize_text(label): hpo_id
        for hpo_id, label in hpo_labels.items()
        if normalize_text(label)
    }


def deduplicate(results: List[ExtractionResult]) -> List[ExtractionResult]:
    """
    Keep the highest-confidence result per HPO ID across all methods.
    Skips structural/metadata HPO terms (HPO_BLOCKLIST).
    """
    best: Dict[str, ExtractionResult] = {}
    for r in results:
        if r.hpo_id in HPO_BLOCKLIST:
            continue
        existing = best.get(r.hpo_id)
        if existing is None or r.confidence > existing.confidence:
            best[r.hpo_id] = r
    return sorted(best.values(), key=lambda x: (x.start or 0, x.hpo_id))


# ── Method 1: Dictionary baseline ────────────────────────────────────────────


def extract_dictionary(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """Exact HPO label matching using regex."""
    normalized = normalize_text(raw_text)
    lookup = build_label_lookup(hpo_labels)
    results = []

    for label_text, hpo_id in lookup.items():
        pattern = rf"\b{re.escape(label_text)}\b"
        for match in re.finditer(pattern, normalized):
            negated = is_negated(normalized, match.start())
            if skip_negated and negated:
                continue
            results.append(ExtractionResult(
                hpo_id=hpo_id,
                label=hpo_labels[hpo_id],
                matched_text=label_text,
                method=ExtractionMethod.DICTIONARY,
                confidence=1.0,
                start=match.start(),
                end=match.end(),
                negated=negated,
            ))

    return results


# ── Method 2: Biomedical NER (d4data) ─────────────────────────────────────────


def extract_biomedical_ner(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
    model_name: str = BIOMEDICAL_NER_MODEL,
) -> List[ExtractionResult]:
    """General biomedical NER using d4data/biomedical-ner-all."""
    try:
        from transformers import pipeline
    except ImportError:
        print("[phenotype] transformers not installed -- skipping biomedical_ner.")
        return []

    ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
    lookup = build_label_lookup(hpo_labels)
    normalized_full = normalize_text(raw_text)
    results = []

    for ent in ner(raw_text):
        span_text = ent["word"].strip()
        normalized_span = normalize_text(span_text)

        hpo_id = lookup.get(normalized_span)
        if not hpo_id:
            for label_norm, candidate_id in lookup.items():
                if len(label_norm) < 10:
                    continue
                if normalized_span in label_norm or label_norm in normalized_span:
                    hpo_id = candidate_id
                    break

        if not hpo_id:
            continue
        if float(ent["score"]) < BIOMEDICAL_NER_MIN_CONFIDENCE:
            continue

        negated = is_negated(normalized_full, ent["start"])
        if skip_negated and negated:
            continue

        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=hpo_labels[hpo_id],
            matched_text=span_text,
            method=ExtractionMethod.BIOMEDICAL_NER,
            confidence=float(ent["score"]),
            start=ent["start"],
            end=ent["end"],
            negated=negated,
        ))

    return results


# ── Method 3: FastHPOCR ───────────────────────────────────────────────────────

_fast_hpo_cr_instance = None  # module-level cache


def _get_fast_hpo_cr() -> Optional[object]:
    """Load (or return cached) FastHPOCR instance."""
    global _fast_hpo_cr_instance
    if _fast_hpo_cr_instance is not None:
        return _fast_hpo_cr_instance

    src = str(_FAST_HPO_CR_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)

    try:
        from IndexHPO import IndexHPO
        from HPOAnnotator import HPOAnnotator
    except ImportError:
        print(
            "[phenotype] FastHPOCR not found -- clone into src/fast_hpo_cr/.\n"
            "  git clone https://github.com/tudorgroza/fast_hpo_cr.git src/fast_hpo_cr"
        )
        return None

    if not _HP_OBO_PATH.exists():
        print(
            f"[phenotype] hp.obo not found at {_HP_OBO_PATH}.\n"
            f"  wget https://purl.obolibrary.org/obo/hp.obo -O {_HP_OBO_PATH}"
        )
        return None

    _FAST_HPO_CR_IDX_DIR.mkdir(parents=True, exist_ok=True)
    index_dir = str(_FAST_HPO_CR_IDX_DIR.resolve())
    obo_path  = str(_HP_OBO_PATH.resolve())

    # FastHPOCR looks for 'resources/' relative to cwd -- must run from its src dir
    original_dir = os.getcwd()
    os.chdir(str(_FAST_HPO_CR_SRC))

    try:
        index_files = list(_FAST_HPO_CR_IDX_DIR.iterdir())
        if not index_files:
            print("[phenotype] Building FastHPOCR index (can take several minutes, first run only)...")
            from IndexHPO import IndexHPO
            IndexHPO(obo_path, index_dir).index()
            print("[phenotype] FastHPOCR index built.")
        else:
            print("[phenotype] FastHPOCR index found, loading...")

        from HPOAnnotator import HPOAnnotator
        _fast_hpo_cr_instance = HPOAnnotator(os.path.join(index_dir, "hp.index"))
        print("[phenotype] FastHPOCR ready.")
    finally:
        os.chdir(original_dir)

    return _fast_hpo_cr_instance


def extract_fast_hpo_cr(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    HPO concept recognition using FastHPOCR.
    Morphologically-equivalent token clusters for robust lexical variability.
    Repo  : https://github.com/tudorgroza/fast_hpo_cr (clone into src/)
    Paper : https://doi.org/10.1093/bioinformatics/btae406
    """
    cr = _get_fast_hpo_cr()
    if cr is None:
        return []

    normalized_full = normalize_text(raw_text)
    results = []

    try:
        annotations = cr.annotate(raw_text)
    except Exception as e:
        print(f"[phenotype] FastHPOCR annotation failed: {e}")
        return []

    for ann in annotations:
        hpo_id  = getattr(ann, "hpoUri", None)
        matched = getattr(ann, "textSpan", "")
        start   = getattr(ann, "startOffset", None)
        end     = getattr(ann, "endOffset", None)

        if not hpo_id:
            continue

        negated = is_negated(normalized_full, start or 0)
        if skip_negated and negated:
            continue

        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=hpo_labels.get(hpo_id, hpo_id),
            matched_text=matched,
            method=ExtractionMethod.FAST_HPO_CR,
            confidence=0.90,
            start=start,
            end=end,
            negated=negated,
        ))

    return results


# ── Method 4: ChatGPT extraction ──────────────────────────────────────────────


def extract_chatgpt(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
    model: str = _CHATGPT_MODEL,
) -> List[ExtractionResult]:
    """
    HPO extraction using GPT-4o-mini.

    The model extracts phenotype phrases from the clinical text. The phrases
    are then mapped locally to official HPO IDs using hpo_labels, so the model
    does not directly generate ontology identifiers.

    Requires: pip install openai
    Add key to env file: OPENAI_API_KEY=sk-...
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[phenotype] OPENAI_API_KEY not set -- skipping chatgpt extraction.\n"
            "  Add key to env file: OPENAI_API_KEY=sk-..."
        )
        return []

    try:
        from openai import OpenAI
    except ImportError:
        print("[phenotype] openai not installed -- skipping chatgpt.\n  pip install openai")
        return []

    client = OpenAI(api_key=api_key)
    results = []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHATGPT_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present.
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"```$", "", content).strip()

        # Extract the JSON object if the model accidentally adds surrounding text.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        parsed = json.loads(content)
        phenotypes = parsed.get("phenotypes", [])

    except Exception as e:
        print(f"[phenotype] ChatGPT extraction failed: {e}")
        return []

    if not isinstance(phenotypes, list):
        return []

    lookup = build_label_lookup(hpo_labels)
    mapped_terms = []
    seen = set()

    for phrase in phenotypes[:20]:
        if not isinstance(phrase, str):
            continue

        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            continue

        hpo_id = lookup.get(phrase_norm)

        # Conservative fallback: allow containment only for reasonably specific
        # multi-word labels/phrases. This improves recall while avoiding broad
        # one-word false matches.
        if not hpo_id and len(phrase_norm) >= 8:
            for label_norm, candidate_id in lookup.items():
                if len(label_norm) < 8:
                    continue
                if phrase_norm == label_norm or phrase_norm in label_norm or label_norm in phrase_norm:
                    hpo_id = candidate_id
                    break

        if not hpo_id:
            continue
        if hpo_id in HPO_BLOCKLIST:
            continue
        if hpo_id in seen:
            continue

        seen.add(hpo_id)
        mapped_terms.append((hpo_id, phrase.strip()))

    for hpo_id, phrase in mapped_terms:
        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=hpo_labels.get(hpo_id, hpo_id),
            matched_text=phrase,
            method=ExtractionMethod.CHATGPT,
            confidence=0.85,
            start=None,
            end=None,
            negated=False,
        ))

    return results


# ── Method 5: PhenoBrain API ──────────────────────────────────────────────────


def extract_phenobrain_api(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    HPO extraction via PhenoBrain's public web API.

    PhenoBrain uses a BERT-based NLP model trained on EHR clinical notes.
    The API is async: submit text -> get task ID -> poll for results.

    API docs: https://github.com/xiaohaomao/timgroup_disease_diagnosis/tree/main/PhenoBrain_Web_API
    No API key required.
    """
    try:
        import requests
    except ImportError:
        print("[phenotype] requests not installed -- skipping phenobrain_api.\n  pip install requests")
        return []

    # Step 1: Submit text via POST
    try:
        resp = requests.post(
            _PHENOBRAIN_SUBMIT_URL,
            json={"text": raw_text, "method": "HPO/CHPO", "threshold": ""},
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("TASK_ID")
        if not task_id:
            print(f"[phenotype] PhenoBrain API: no TASK_ID in response: {resp.json()}")
            return []
    except Exception as e:
        print(f"[phenotype] PhenoBrain API submit failed: {e}")
        return []

    # Step 2: Poll for results
    hpo_list = []
    hpo_to_info = {}
    for _ in range(_PHENOBRAIN_MAX_POLLS):
        time.sleep(_PHENOBRAIN_POLL_INTERVAL)
        try:
            result_resp = requests.get(
                _PHENOBRAIN_RESULT_URL,
                params={"taskId": task_id},
                timeout=30,
            )
            result_resp.raise_for_status()
            data = result_resp.json()
            state = data.get("state", "")

            if state == "SUCCESS":
                result = data.get("result", {})
                hpo_list = result.get("HPO_LIST", [])
                hpo_to_info = result.get("HPO_TO_INFO", {})
                break
            elif state in ("PROCESS_TEXT", "EXTRACT_HPO"):
                continue
            else:
                print(f"[phenotype] PhenoBrain API unexpected state: {state}")
                return []
        except Exception as e:
            print(f"[phenotype] PhenoBrain API poll failed: {e}")
            return []
    else:
        print("[phenotype] PhenoBrain API timed out waiting for results.")
        return []

    # Step 3: Convert to ExtractionResult
    results = []
    seen = set()
    for hpo_id in hpo_list:
        if not hpo_id or hpo_id in HPO_BLOCKLIST or hpo_id in seen:
            continue
        seen.add(hpo_id)
        info = hpo_to_info.get(hpo_id, {})
        label = info.get("ENG_NAME") or hpo_labels.get(hpo_id, hpo_id)

        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=label,
            matched_text=hpo_id,
            method=ExtractionMethod.PHENOBRAIN_API,
            confidence=0.85,
            start=None,
            end=None,
            negated=False,
        ))

    return results


# ── Ensemble entry point ──────────────────────────────────────────────────────


def extract_hpo_terms(
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] = ("dictionary",),
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Run one or more extraction methods and merge results.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID to label string.
        methods:       Subset of:
                         "dictionary"     -- exact label matching
                         "biomedical_ner" -- d4data transformer NER
                         "fast_hpo_cr"    -- FastHPOCR (clone into src/)
                         "chatgpt"        -- GPT-4o-mini extraction
                         "phenobrain_api" -- PhenoBrain public API
        skip_negated:  If True, skip negated mentions (e.g. "no ataxia").

    Returns:
        Deduplicated list of ExtractionResult, sorted by position.
    """
    all_results: List[ExtractionResult] = []

    if "dictionary" in methods:
        all_results += extract_dictionary(raw_text, hpo_labels, skip_negated)

    if "biomedical_ner" in methods:
        all_results += extract_biomedical_ner(raw_text, hpo_labels, skip_negated)

    if "fast_hpo_cr" in methods:
        all_results += extract_fast_hpo_cr(raw_text, hpo_labels, skip_negated)

    if "chatgpt" in methods:
        all_results += extract_chatgpt(raw_text, hpo_labels, skip_negated)

    if "phenobrain_api" in methods:
        all_results += extract_phenobrain_api(raw_text, hpo_labels, skip_negated)

    return deduplicate(all_results)


# ── Patient profile builder ───────────────────────────────────────────────────


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] = ("dictionary",),
) -> Tuple[dict, List[dict]]:
    """
    Build a patient profile dict from raw clinical text.

    Returns:
        patient:         Dict with patient_id, raw_text, hpo_terms,
                         propagated_hpo_terms, methods_used.
        extracted_terms: List of dicts with full extraction provenance.
    """
    extracted = extract_hpo_terms(
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
    )

    hpo_terms = sorted({r.hpo_id for r in extracted})

    # Compute propagated terms using HPO ancestor hierarchy
    try:
        ancestors = load_json(HPO_ANCESTORS_PATH)
        ancestor_sets = preprocess_ancestor_sets(ancestors)
        propagated = set()
        for term in hpo_terms:
            propagated |= get_ancestors_inclusive(term, ancestor_sets)
        propagated_hpo_terms = sorted(propagated)
    except Exception as e:
        print(f"[phenotype] Warning: could not compute propagated terms: {e}")
        propagated_hpo_terms = hpo_terms

    patient = {
        "patient_id": patient_id,
        "raw_text": raw_text,
        "hpo_terms": hpo_terms,
        "propagated_hpo_terms": propagated_hpo_terms,
        "methods_used": list(methods),
    }

    return patient, [r.to_dict() for r in extracted]


# ── Pipeline entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Run phenotype extraction on the example patient."""
    PHENOTYPE_DIR.mkdir(parents=True, exist_ok=True)

    hpo_labels = load_json(HPO_LABELS_PATH)

    patient_data = load_json(PATIENT_PATH)
    raw_text = patient_data.get("raw_text", "").strip()
    patient_id = patient_data.get("patient_id", "patient_001")

    print(f"Running phenotype extraction with methods: {EXTRACTION_METHODS}\n")

    patient, extracted_terms = build_patient_profile(
        patient_id=patient_id,
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=EXTRACTION_METHODS,
    )

    save_json(patient, OUTPUT_PATIENT_PATH)
    save_json(extracted_terms, OUTPUT_EXTRACTION_PATH)

    print(f"Extracted {len(extracted_terms)} HPO term(s):\n")
    for row in extracted_terms:
        print(
            f"  {row['hpo_id']} | {row['label']:<40} | "
            f"conf={row['confidence']:.2f} | "
            f"method={row['method']}"
        )

    print(f"\nPatient profile   -> {OUTPUT_PATIENT_PATH}")
    print(f"Extraction detail -> {OUTPUT_EXTRACTION_PATH}")


if __name__ == "__main__":
    main()
