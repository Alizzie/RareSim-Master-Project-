"""
Shared text preprocessing utilities for similarity pipelines.

Any method that operates on clinical free text, TF-IDF, Word2Vec over
descriptions, LLM preprocessing, should import from here rather than
reimplementing tokenization or document extraction.

This module handles raw text -> token conversion only.
Weighting (IDF, TFxIDF, embeddings) is method-specific and stays in
each method's own module.
"""

import re
from collections import Counter
from typing import Any

# ── Stopwords ─────────────────────────────────────────────────────────────────

# General English function words plus clinical filler phrases that appear in
# almost every disease description and carry no discriminative signal.
# Methods that need a different stopword set can extend this via
# get_stopwords() rather than reimporting and modifying directly.

_BASE_STOPWORDS: set[str] = {
    # English function words
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "not",
    "no",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
    "also",
    "however",
    "therefore",
    # Clinical filler — present in almost every disease description
    "patient",
    "patients",
    "disease",
    "disorder",
    "syndrome",
    "condition",
    "rare",
    "characterized",
    "characterised",
    "including",
    "include",
    "associated",
    "association",
    "presents",
    "present",
    "clinical",
    "features",
    "feature",
    "signs",
    "sign",
    "symptoms",
    "symptom",
    "type",
    "types",
    "form",
    "forms",
    "onset",
    "usually",
    "often",
    "typically",
    "reported",
    "cases",
    "case",
    "affected",
    "individuals",
    "individual",
    "common",
    "commonly",
    "due",
    "caused",
}


def get_stopwords(extra: set[str] | None = None) -> set[str]:
    """
    Return the base stopword set, optionally extended with method-specific words.

    Args:
        extra: Additional tokens to treat as stopwords for a specific method.

    Returns:
        Union of base stopwords and extra (if provided).
    """
    if extra:
        return _BASE_STOPWORDS | extra
    return _BASE_STOPWORDS


# ── Tokenizer ─────────────────────────────────────────────────────────────────


def tokenize(
    text: str,
    min_length: int = 3,
    stopwords: set[str] | None = None,
) -> list[str]:
    """
    Tokenize a clinical text string into normalized tokens.

    Steps:
        1. Lowercase the full string.
        2. Extract alphabetic runs (splits on digits, punctuation, hyphens).
        3. Remove stopwords.
        4. Remove tokens shorter than min_length.

    Args:
        text:       Raw clinical text (patient note or disease description).
        min_length: Minimum token character length to retain. Default 3
                    removes single letters and two-letter noise tokens.
        stopwords:  Token set to exclude. Defaults to _BASE_STOPWORDS.
                    Pass get_stopwords(extra={...}) to extend.

    Returns:
        List of normalized tokens. Order is preserved so callers that
        need positional information can use it.
    """
    if not text:
        return []

    sw = stopwords if stopwords is not None else _BASE_STOPWORDS
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) >= min_length and t not in sw]


def tokenize_to_set(
    text: str,
    min_length: int = 3,
    stopwords: set[str] | None = None,
) -> set[str]:
    """
    Tokenize text and return the unique token set (no duplicates).
    """
    return set(tokenize(text, min_length=min_length, stopwords=stopwords))


# ── Frequency vectors ─────────────────────────────────────────────────────────


def build_token_counts(tokens: list[str]) -> dict[str, float]:
    """
    Build a raw term-frequency vector from a token list.

    Uses raw counts because downstream cosine normalization
    handles scale differences between documents.

    Args:
        tokens: Output of tokenize().

    Returns:
        Dict mapping token -> count as float (compatible with sparse
        vector functions in utils/math.py).
    """
    return {token: float(count) for token, count in Counter(tokens).items()}


def text_to_token_counts(
    text: str,
    min_length: int = 3,
    stopwords: set[str] | None = None,
) -> dict[str, float]:
    """
    Convert a raw text string to a token count vector in one step.
    Convenience wrapper: tokenize -> count.

    Args:
        text:       Raw clinical text.
        min_length: Passed to tokenize().
        stopwords:  Passed to tokenize().

    Returns:
        Dict mapping token -> float count.
    """
    return build_token_counts(
        tokenize(text, min_length=min_length, stopwords=stopwords)
    )


# ── Document extractors ───────────────────────────────────────────────────────


def get_disease_text(
    profile: dict,
    text_field: str = "merged_description",
) -> tuple[str, bool]:
    """
    Extract the text document for a disease profile.

    Falls back to label if the preferred field is empty.

    Returns:
        (text, used_fallback) — used_fallback is True when the label
        was used instead of the description. Callers can surface this
        in diagnostics so users know a short token vector is a data
        coverage issue, not a real property of the disease.
    """
    text = (profile.get(text_field) or "").strip()
    if text:
        return text, False
    label = (profile.get("label") or "").strip()
    return label, True


def get_patient_text(patient_raw_text: str) -> str:
    """
    Return the patient's raw clinical note, stripped.

    Kept as a named function for symmetry with get_disease_text and to
    allow future preprocessing (section extraction, de-identification)
    without touching any pipeline code.
    """
    return (patient_raw_text or "").strip()


# ── HPO label tokenization ────────────────────────────────────────────────────


def hpo_terms_to_token_counts(
    hpo_terms: set[str],
    hpo_labels: dict[str, str],
    min_length: int = 3,
    stopwords: set[str] | None = None,
) -> dict[str, float]:
    """
    Convert a set of HPO terms to a token count vector via their labels.

    Useful for hybrid methods that match patient HPO label words against
    disease description text, bridging the structured and free-text worlds.

    Example:
        {"HP:0001251"} with label "Ataxia"
            → {"ataxia": 1.0}

        {"HP:0002187"} with label "Intellectual disability, profound"
            → {"intellectual": 1.0, "disability": 1.0, "profound": 1.0}

    Args:
        hpo_terms:  Set of HPO IDs.
        hpo_labels: HPO ID → human-readable label.
        min_length: Passed to tokenize().
        stopwords:  Passed to tokenize(). Defaults to base stopwords.

    Returns:
        Token count dict aggregated across all HPO label strings.
    """
    all_tokens: list[str] = []
    for term in hpo_terms:
        label = hpo_labels.get(term, "")
        all_tokens.extend(tokenize(label, min_length=min_length, stopwords=stopwords))
    return build_token_counts(all_tokens)
