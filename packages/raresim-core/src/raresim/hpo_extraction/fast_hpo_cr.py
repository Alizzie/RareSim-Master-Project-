"""
FastHPOCR extractor — morphological token cluster dictionary matching.

Uses morphologically-equivalent token clusters for robust lexical variability.
Significantly better recall than plain dictionary matching for clinical text.

Setup:
    git clone https://github.com/tudorgroza/fast_hpo_cr.git third_party/fast_hpo_cr

Paper: https://doi.org/10.1093/bioinformatics/btae406
"""

import importlib
import os
import sys
from functools import cache
from typing import Any, Protocol, cast

from raresim.utils.paths import FAST_HPO_CR_DIR, ONTOLOGY_DIR, OUTPUTS_DIR

from raresim.hpo_extraction._types import ExtractionMethod, ExtractionResult
from raresim.hpo_extraction._utils import is_negated, normalize_text


class FastHPOCRAnnotator(Protocol):  # pylint: disable=too-few-public-methods
    """Minimal interface used from the FastHPOCR annotator."""

    def annotate(self, text: str) -> list[Any]:
        """Return FastHPOCR annotations for a text."""
        raise NotImplementedError


# ── Paths ──────────────────────────────────────────────────────────────────────

_HP_OBO_PATH = ONTOLOGY_DIR / "hpo.obo"
_FAST_HPO_CR_IDX_DIR = OUTPUTS_DIR / "fast_hpo_cr_index"


def _ensure_fast_hpo_cr_on_path() -> None:
    """Make FastHPOCR importable from the local third-party directory."""
    src = str(FAST_HPO_CR_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


@cache
def _get_fast_hpo_cr() -> FastHPOCRAnnotator | None:
    """Load and cache the FastHPOCR annotator instance."""
    _ensure_fast_hpo_cr_on_path()

    try:
        index_module = importlib.import_module("IndexHPO")
        annotator_module = importlib.import_module("HPOAnnotator")
    except ImportError:
        print(
            "[fast_hpo_cr] FastHPOCR not found -- clone into third_party/fast_hpo_cr.\n"
            "  git clone https://github.com/tudorgroza/fast_hpo_cr.git "
            "third_party/fast_hpo_cr"
        )
        return None

    index_hpo_cls = getattr(index_module, "IndexHPO")
    hpo_annotator_cls = getattr(annotator_module, "HPOAnnotator")

    _FAST_HPO_CR_IDX_DIR.mkdir(parents=True, exist_ok=True)
    index_dir = str(_FAST_HPO_CR_IDX_DIR.resolve())
    obo_path = str(_HP_OBO_PATH.resolve())

    original_dir = os.getcwd()
    os.chdir(str(FAST_HPO_CR_DIR))

    try:
        index_files = list(_FAST_HPO_CR_IDX_DIR.iterdir())
        if not index_files:
            print(
                "[fast_hpo_cr] Building index "
                "(first run only, may take several minutes)..."
            )
            index_hpo_cls(obo_path, index_dir).index()
            print("[fast_hpo_cr] Index built.")
        else:
            print("[fast_hpo_cr] Index found, loading...")

        annotator = hpo_annotator_cls(os.path.join(index_dir, "hp.index"))
        print("[fast_hpo_cr] Ready.")
        return cast(FastHPOCRAnnotator, annotator)
    finally:
        os.chdir(original_dir)


def extract_fast_hpo_cr(  # pylint: disable=too-many-locals
    raw_text: str,
    hpo_labels: dict[str, str],
    skip_negated: bool = True,
) -> list[ExtractionResult]:
    """
    HPO concept recognition using FastHPOCR.

    Args:
        raw_text: Raw clinical patient text.
        hpo_labels: Dict mapping HPO ID to label string.
        skip_negated: If True, skip negated mentions.

    Returns:
        Extraction results for each annotated concept.
    """
    cr = _get_fast_hpo_cr()
    if cr is None:
        return []

    normalized_full = normalize_text(raw_text)
    results: list[ExtractionResult] = []

    try:
        annotations = cr.annotate(raw_text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[fast_hpo_cr] Annotation failed: {exc}")
        return []

    for ann in annotations:
        hpo_id_value = getattr(ann, "hpoUri", None)
        matched_value = getattr(ann, "textSpan", "")
        start_value = getattr(ann, "startOffset", None)
        end_value = getattr(ann, "endOffset", None)

        if not isinstance(hpo_id_value, str) or not hpo_id_value:
            continue

        hpo_id = hpo_id_value
        label = hpo_labels.get(hpo_id) or hpo_id

        matched = matched_value if isinstance(matched_value, str) else ""
        start = start_value if isinstance(start_value, int) else None
        end = end_value if isinstance(end_value, int) else None

        negated = is_negated(normalized_full, start if start is not None else 0)
        if skip_negated and negated:
            continue

        results.append(
            ExtractionResult(
                hpo_id=hpo_id,
                label=label,
                matched_text=matched,
                method=ExtractionMethod.FAST_HPO_CR,
                confidence=0.90,
                start=start,
                end=end,
                negated=negated,
            )
        )

    return results
