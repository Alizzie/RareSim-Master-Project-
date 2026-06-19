"""
TF-IDF vector computation for all three TF-IDF modes.
"""

import math

from raresim.utils.text_utils import (
    text_to_token_counts,
    get_disease_text,
    hpo_terms_to_token_counts,
)

from raresim.similarity_methods.tfidf.config import DISEASE_TEXT_FIELD


def compute_idf(
    disease_profiles: dict[str, dict],
    propagated_term_key: str = "propagated_hpo_terms",
) -> dict[str, float]:
    """
    Compute IDF over the disease corpus using HPO term presence.

    IDF(term) = log(N / df(term))
        N      = total number of disease profiles
        df     = number of profiles containing the term

    Binary TF means IDF alone carries the weighting signal:
        - Terms in every disease: IDF ≈ 0 -> no discriminative signal
        - Terms in 1 of 10 000 diseases: high IDF -> strong signal

    Uses propagated terms by default so ancestor terms are included,
    which naturally lowers their IDF (they appear in more diseases).
    """

    total = len(disease_profiles)
    doc_freq: dict[str, int] = {}

    for profile in disease_profiles.values():
        for term in set(profile.get(propagated_term_key, [])):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    # IDF: rare terms (low doc_freq) get high scores, common terms get low scores
    return {term: math.log(total / freq) for term, freq in doc_freq.items()}


def build_tfidf_vector(
    terms: set[str],
    idf: dict[str, float],
) -> dict[str, float]:
    """
    Build a binary TF-IDF vector for a set of HPO terms.

    Weight = 1 (binary TF) x IDF = IDF value for each present term.
    Terms not seen in the IDF corpus are ignored (no unseen-term penalty).
    """
    return {term: idf[term] for term in terms if term in idf}


# ── Text mode ─────────────────────────────────────────────────────────────────


def compute_text_idf(
    disease_profiles: dict[str, dict],
    text_field: str = DISEASE_TEXT_FIELD,
) -> dict[str, float]:
    """
        Compute IDF over the disease corpus using tokenized description text.

    Each disease profile's description is tokenized and the document
        frequency of each token is counted across all profiles.

        Args:
            disease_profiles: Disease profile dicts.
            text_field:       Profile field to use as the disease document.

        Returns:
            Dict mapping token -> IDF value.
    """
    total = len(disease_profiles)
    doc_freq: dict[str, int] = {}

    for profile in disease_profiles.values():
        text, _ = get_disease_text(profile, text_field=text_field)

        if not text:
            continue

        for token in set(text_to_token_counts(text).keys()):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    return {
        token: math.log(total / freq) for token, freq in doc_freq.items() if freq > 0
    }


def build_text_tfidf_vector(
    tf_vector: dict[str, float],
    idf: dict[str, float],
) -> dict[str, float]:
    """
    Build a TF-IDF vector from a raw TF vector and an IDF dict.

    Unlike the HPO mode, TF here is a real count (not binary), so
    the weight is TF x IDF rather than just IDF.

    Args:
        tf_vector: Raw term-frequency dict {token: count}.
        idf:       IDF dict {token: idf_value}.

    Returns:
        Sparse TF-IDF vector {token: tf_idf_weight}.
    """
    return {token: tf * idf[token] for token, tf in tf_vector.items() if token in idf}


def build_disease_text_vector(
    profile: dict[str, dict],
    idf: dict[str, float],
    text_field: str = DISEASE_TEXT_FIELD,
) -> tuple[dict[str, float], bool]:
    """
    Build a TF-IDF vector for a disease profile from its description text.
    Convenience wrapper: extract text -> TF -> TF-IDF.
    """

    text, used_fallback = get_disease_text(profile, text_field=text_field)

    if not text:
        return {}, used_fallback

    tf_vector = text_to_token_counts(text)
    return build_text_tfidf_vector(tf_vector, idf), used_fallback


def build_patient_text_vector(
    raw_text: str,
    idf: dict[str, float],
) -> dict[str, float]:
    """
    Build a TF-IDF vector for a patient from their raw text.
    Convenience wrapper: tokenize -> TF -> TF-IDF.
    """

    if not raw_text:
        return {}

    tf_vector = text_to_token_counts(raw_text)
    return build_text_tfidf_vector(tf_vector, idf)


# ── Hybrid mode ───────────────────────────────────────────────────────────────


def build_patient_hybrid_vector(
    hpo_terms: set[str],
    hpo_labels: dict[str, str],
    idf: dict[str, float],
) -> dict[str, float]:
    """
    Build a TF-IDF vector for the patient using HPO label tokens.

    Converts each HPO term to its label string, tokenizes, then weights
    by the text-mode IDF. Allows matching patient HPO labels against
    disease description text.

    Args:
        hpo_terms:  Patient HPO term IDs.
        hpo_labels: HPO ID -> label.
        idf:        Text-mode IDF (from compute_text_idf).
    """

    if not hpo_terms:
        return {}

    tf_vector = hpo_terms_to_token_counts(hpo_terms, hpo_labels)
    return build_text_tfidf_vector(tf_vector, idf)
