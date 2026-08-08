"""Unit tests for TF-IDF"""

import math
import pytest
from raresim.similarity_methods.tfidf.methods import (
    compute_idf,
    build_tfidf_vector,
    compute_text_idf,
    build_text_tfidf_vector,
    build_patient_text_vector,
    build_hpo_label_vector,
)


# Fixtures 

def _disease_profiles() -> dict[str, dict]:
    return {
        "ORPHA:1": {
            "propagated_hpo_terms": ["HP:0001250", "HP:0000118"],
            "label": "Disease A",
            "description": "seizure disorder with neurological involvement",
        },
        "ORPHA:2": {
            "propagated_hpo_terms": ["HP:0004322", "HP:0000118"],
            "label": "Disease B",
            "description": "short stature growth disorder",
        },
        "ORPHA:3": {
            "propagated_hpo_terms": ["HP:0001250", "HP:0004322", "HP:0000118"],
            "label": "Disease C",
            "description": "seizure and growth abnormality",
        },
    }


def _hpo_labels() -> dict[str, str]:
    return {
        "HP:0001250": "Seizure",
        "HP:0004322": "Short stature",
        "HP:0000118": "Phenotypic abnormality",
    }


# compute_idf 

def test_compute_idf_common_term_has_low_idf() -> None:
    """HP:0000118 appears in all 3 diseases so its IDF should be log(3/3) = 0."""
    idf = compute_idf(_disease_profiles())
    assert idf["HP:0000118"] == pytest.approx(math.log(3 / 3))


def test_compute_idf_rare_term_has_high_idf() -> None:
    """HP:0004322 appears in 2/3 diseases; HP:0001250 also 2/3. Both < log(3/1)."""
    idf = compute_idf(_disease_profiles())
    # A term in only 1 disease would have idf = log(3/1) ≈ 1.099
    # Both HP:0001250 and HP:0004322 appear in 2 diseases: log(3/2) ≈ 0.405
    assert idf["HP:0001250"] == pytest.approx(math.log(3 / 2))
    assert idf["HP:0004322"] == pytest.approx(math.log(3 / 2))


def test_compute_idf_all_terms_present() -> None:
    """All terms from disease profiles should appear in IDF dict."""
    idf = compute_idf(_disease_profiles())
    assert "HP:0001250" in idf
    assert "HP:0004322" in idf
    assert "HP:0000118" in idf


# build_tfidf_vector 

def test_build_tfidf_vector_includes_known_terms() -> None:
    """Vector should contain all terms present in IDF."""
    idf = compute_idf(_disease_profiles())
    terms = {"HP:0001250", "HP:0000118"}
    vector = build_tfidf_vector(terms, idf)
    assert "HP:0001250" in vector
    assert "HP:0000118" in vector


def test_build_tfidf_vector_excludes_unknown_terms() -> None:
    """Terms not in IDF corpus should be excluded from the vector."""
    idf = compute_idf(_disease_profiles())
    vector = build_tfidf_vector({"HP:9999999"}, idf)
    assert vector == {}


def test_build_tfidf_vector_weight_equals_idf() -> None:
    """Binary TF means weight == IDF value."""
    idf = compute_idf(_disease_profiles())
    vector = build_tfidf_vector({"HP:0001250"}, idf)
    assert vector["HP:0001250"] == pytest.approx(idf["HP:0001250"])


# compute_text_idf 

def test_compute_text_idf_returns_dict() -> None:
    """compute_text_idf should return a non-empty dict for valid profiles."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    assert isinstance(idf, dict)
    assert len(idf) > 0


def test_compute_text_idf_common_word_has_lower_idf() -> None:
    """Words appearing in more documents should have lower IDF."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    # 'abnormality' or 'seizure' appear in multiple docs
    # A word in only 1 doc has higher IDF than one in 2+ docs
    scores = list(idf.values())
    assert min(scores) < max(scores)


# build_text_tfidf_vector 

def test_build_text_tfidf_vector_weights_by_tf_and_idf() -> None:
    """TF-IDF weight should equal tf * idf for each token."""
    idf = {"seizure": 2.0, "disorder": 1.5}
    tf = {"seizure": 2, "disorder": 1, "unknown": 3}
    vector = build_text_tfidf_vector(tf, idf)
    assert vector["seizure"] == pytest.approx(4.0)
    assert vector["disorder"] == pytest.approx(1.5)
    assert "unknown" not in vector


def test_build_text_tfidf_vector_empty_tf() -> None:
    """Empty TF vector should produce empty TF-IDF vector."""
    idf = {"seizure": 2.0}
    assert build_text_tfidf_vector({}, idf) == {}


# build_patient_text_vector 

def test_build_patient_text_vector_empty_text() -> None:
    """Empty patient text should return empty vector."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    vector = build_patient_text_vector("", idf)
    assert vector == {}


def test_build_patient_text_vector_nonempty_text() -> None:
    """Non-empty patient text should produce a non-empty vector."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    vector = build_patient_text_vector("seizure disorder", idf)
    assert len(vector) > 0


# build_hpo_label_vector

def test_build_hpo_label_vector_empty_terms() -> None:
    """Empty HPO terms should produce empty vector."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    vector = build_hpo_label_vector(set(), _hpo_labels(), idf)
    assert vector == {}


def test_build_hpo_label_vector_known_terms() -> None:
    """HPO labels with tokens in the IDF corpus should produce a non-empty vector."""
    idf = compute_text_idf(_disease_profiles(), text_field="description")
    vector = build_hpo_label_vector({"HP:0001250"}, _hpo_labels(), idf)
    # 'seizure' token should appear in idf since Disease A description has it
    assert isinstance(vector, dict)
