"""Methods for computing TF-IDF vectors."""

import math


def compute_idf(
    disease_profiles: dict[str, dict],
    propagated_term_key: str = "propagated_hpo_terms",
) -> dict[str, float]:
    """
    Measures how rare or common a term is across the entire corpus of disease profiles.
    IDF(term) = log(total_diseases / number_of_diseases_containing_term)
    Binary TF means IDF alone carries the weighting signal.
    - Terms in every disease -> IFD = log(1) = 0 -> no discriminative signal
    - Terms in 1/10000 diseases -> high IDF -> strong signal

    Propagated terms (true-path) are used by default so ancestor terms are included,
    which naturally lowers their IDF (aka they appear in more diseases).
    """

    total = len(disease_profiles)

    # Count in how many disease profiles each HPO term appears
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
    Build a sparse TF-IDF vector for a set of HPO terms.
    Binary TF (1 if present) × IDF = just the IDF value for each present term.
    Terms not seen during IDF computation (e.g. unseen in any disease) are ignored.
    """
    return {term: idf[term] for term in terms if term in idf}
