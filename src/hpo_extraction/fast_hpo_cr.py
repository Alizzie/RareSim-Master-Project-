"""
FastHPOCR extractor — morphological token cluster dictionary matching.

Uses morphologically-equivalent token clusters for robust lexical variability.
Significantly better recall than plain dictionary matching for clinical text.

Setup:
    git clone https://github.com/tudorgroza/fast_hpo_cr.git src/fast_hpo_cr
    wget https://purl.obolibrary.org/obo/hp.obo -O ontologies/model/hp.obo

Paper: https://doi.org/10.1093/bioinformatics/btae406
"""

import os
import sys
from typing import Dict, List, Optional

from shared.paths import OUTPUTS_DIR, PROJECT_ROOT

from ._types import ExtractionMethod, ExtractionResult
from ._utils import is_negated, normalize_text

# ── Paths ──────────────────────────────────────────────────────────────────────

_FAST_HPO_CR_SRC     = PROJECT_ROOT / "src" / "fast_hpo_cr"
_HP_OBO_PATH         = PROJECT_ROOT / "ontologies" / "model" / "hp.obo"
_FAST_HPO_CR_IDX_DIR = OUTPUTS_DIR / "fast_hpo_cr_index"

# ── Module-level instance cache ────────────────────────────────────────────────

_fast_hpo_cr_instance = None


def _get_fast_hpo_cr() -> Optional[object]:
    """Load (or return cached) FastHPOCR annotator instance."""
    global _fast_hpo_cr_instance
    if _fast_hpo_cr_instance is not None:
        return _fast_hpo_cr_instance

    src = str(_FAST_HPO_CR_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)

    try:
        from IndexHPO import IndexHPO        # noqa: F401
        from HPOAnnotator import HPOAnnotator  # noqa: F401
    except ImportError:
        print(
            "[fast_hpo_cr] FastHPOCR not found -- clone into src/fast_hpo_cr/.\n"
            "  git clone https://github.com/tudorgroza/fast_hpo_cr.git src/fast_hpo_cr"
        )
        return None

    if not _HP_OBO_PATH.exists():
        print(
            f"[fast_hpo_cr] hp.obo not found at {_HP_OBO_PATH}.\n"
            f"  wget https://purl.obolibrary.org/obo/hp.obo -O {_HP_OBO_PATH}"
        )
        return None

    _FAST_HPO_CR_IDX_DIR.mkdir(parents=True, exist_ok=True)
    index_dir = str(_FAST_HPO_CR_IDX_DIR.resolve())
    obo_path  = str(_HP_OBO_PATH.resolve())

    # FastHPOCR looks for 'resources/' relative to cwd — must run from its src dir
    original_dir = os.getcwd()
    os.chdir(str(_FAST_HPO_CR_SRC))

    try:
        index_files = list(_FAST_HPO_CR_IDX_DIR.iterdir())
        if not index_files:
            print("[fast_hpo_cr] Building index (first run only, may take several minutes)...")
            from IndexHPO import IndexHPO
            IndexHPO(obo_path, index_dir).index()
            print("[fast_hpo_cr] Index built.")
        else:
            print("[fast_hpo_cr] Index found, loading...")

        from HPOAnnotator import HPOAnnotator
        _fast_hpo_cr_instance = HPOAnnotator(os.path.join(index_dir, "hp.index"))
        print("[fast_hpo_cr] Ready.")
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

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        skip_negated:  If True, skip negated mentions.

    Returns:
        List of ExtractionResult for each annotated concept.
    """
    cr = _get_fast_hpo_cr()
    if cr is None:
        return []

    normalized_full = normalize_text(raw_text)
    results = []

    try:
        annotations = cr.annotate(raw_text)
    except Exception as e:
        print(f"[fast_hpo_cr] Annotation failed: {e}")
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
    