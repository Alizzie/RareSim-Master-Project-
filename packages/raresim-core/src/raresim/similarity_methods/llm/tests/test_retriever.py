"""Tests for LLM disease retrieval."""

# pylint: disable=protected-access,too-few-public-methods

from typing import Any, cast

from pytest import MonkeyPatch

from raresim.similarity_methods.llm import retriever as llm_retriever
from raresim.types.result import SimilarityResult
from raresim.types.schemas import PatientProfile


MODEL_NAME = "fake-llm-model"


class FakeContext:
    """Fake AppContext-like object."""

    hpo_labels = {"HP:0001250": "Seizure"}
    disease_profiles = {"ORPHA:1": {"label": "Disease A", "hpo_terms": ["HP:0001250"]}}
    ic_values = {"HP:0001250": 5.0}
    disease_ancestors: dict[str, list[str]] = {}
    disease_metadata_index: dict[str, dict[str, Any]] = {}


class FakePipe:
    """Fake HuggingFace text-generation pipeline."""


class FakeExplanationPipe:
    """Fake explanation pipeline that echoes prompts."""

    def __init__(self) -> None:
        """Initialize captured prompts."""
        self.prompts: list[str] = []

    def __call__(
        self,
        prompts: list[str],
        max_new_tokens: int,
    ) -> list[list[dict[str, str]]]:
        """Return fake generated explanations."""
        _ = max_new_tokens
        self.prompts = prompts
        return [
            [{"generated_text": f"{prompt}\nExplanation text."}]
            for prompt in prompts
        ]


class FailingPipe:
    """Fake pipe that should not be called."""

    def __call__(
        self,
        prompts: list[str],
        max_new_tokens: int,
    ) -> list[list[dict[str, str]]]:
        """Raise if called unexpectedly."""
        _ = prompts
        _ = max_new_tokens
        raise AssertionError("Pipe should not have been called.")


def _patient() -> PatientProfile:
    """Build a small patient profile."""
    return PatientProfile(
        patient_id="patient-1",
        raw_text="Patient has seizure.",
        hpo_terms={"HP:0001250"},
        propagated_hpo_terms={"HP:0001250", "HP:0000118"},
    )


def _retriever() -> llm_retriever.LlmDiseaseRetriever:
    """Build an LLM retriever with tiny fake data."""
    return llm_retriever.LlmDiseaseRetriever(
        patient=_patient(),
        hpo_labels={"HP:0001250": "Seizure"},
        disease_profiles={
            "ORPHA:1": {
                "label": "Disease A",
                "hpo_terms": ["HP:0001250"],
            }
        },
        ic_values={"HP:0001250": 5.0},
        disease_ancestors={},
        disease_metadata_index={},
    )


def test_from_context_copies_context_fields() -> None:
    """Test that from_context copies fields from an AppContext-like object."""
    retriever = llm_retriever.LlmDiseaseRetriever.from_context(
        patient=_patient(),
        context=FakeContext(),  # type: ignore[arg-type]
    )

    assert retriever.hpo_labels == FakeContext.hpo_labels
    assert retriever.disease_profiles == FakeContext.disease_profiles
    assert retriever.ic_values == FakeContext.ic_values


def test_retrieve_with_pipe_queries_and_parses_model_output(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that retrieve_with_pipe builds a prompt, queries, and parses output."""
    captured: dict[str, Any] = {}

    def fake_build_retrieval_prompt(
        patient: PatientProfile,
        hpo_labels: dict[str, str],
        top_k: int,
    ) -> str:
        """Return a deterministic retrieval prompt."""
        captured["patient"] = patient
        captured["hpo_labels"] = hpo_labels
        captured["top_k"] = top_k
        return "retrieval prompt"

    def fake_query_hf(
        prompt: str,
        pipe: FakePipe,
        max_tokens: int,
    ) -> str:
        """Return fake LLM output."""
        captured["prompt"] = prompt
        captured["pipe"] = pipe
        captured["max_tokens"] = max_tokens
        return "ORPHA:1 Disease A"

    def fake_parse_retrieval_output(**kwargs: Any) -> list[SimilarityResult]:
        """Return fake parsed SimilarityResult objects."""
        captured["generated_text"] = kwargs["generated_text"]
        captured["model_name"] = kwargs["model_name"]
        captured["top_k_parse"] = kwargs["top_k"]
        return [
            SimilarityResult(
                disease_id="ORPHA:1",
                label="Disease A",
                score=1.0,
                method_name=MODEL_NAME,
                rank=1,
                explanation={
                    "diagnostics": {
                        "validated_against_profiles": True,
                    }
                },
            )
        ]

    monkeypatch.setattr(
        llm_retriever,
        "build_retrieval_prompt",
        fake_build_retrieval_prompt,
    )
    monkeypatch.setattr(llm_retriever, "query_hf", fake_query_hf)
    monkeypatch.setattr(
        llm_retriever,
        "parse_retrieval_output",
        fake_parse_retrieval_output,
    )

    pipe = FakePipe()
    results = _retriever().retrieve_with_pipe(
        pipe=pipe,
        model_name=MODEL_NAME,
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].disease_id == "ORPHA:1"
    assert captured["prompt"] == "retrieval prompt"
    assert captured["pipe"] is pipe
    assert captured["generated_text"] == "ORPHA:1 Disease A"
    assert captured["model_name"] == MODEL_NAME
    assert captured["top_k"] == 3
    assert captured["top_k_parse"] == 3


def test_explain_results_with_pipe_adds_clinical_explanation(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that candidate results receive LLM-generated explanation fields."""

    def fake_patient_context_text(
        patient: PatientProfile,
        hpo_labels: dict[str, str],
    ) -> str:
        """Return fake patient context text."""
        _ = patient
        _ = hpo_labels
        return "patient context"

    def fake_explanation_prompt(**_kwargs: Any) -> str:
        """Return a deterministic explanation prompt."""
        return "explanation prompt"

    def fake_parse_explanation(_generated: str) -> dict[str, str]:
        """Return parsed explanation fields."""
        return {
            "text": "Parsed clinical explanation.",
            "verdict": "possible",
            "verdict_reason": "The disease shares seizure.",
        }

    monkeypatch.setattr(
        llm_retriever,
        "build_patient_context_text",
        fake_patient_context_text,
    )
    monkeypatch.setattr(
        llm_retriever,
        "build_explanation_prompt",
        fake_explanation_prompt,
    )
    monkeypatch.setattr(llm_retriever, "parse_explanation", fake_parse_explanation)

    candidate = SimilarityResult(
        disease_id="ORPHA:1",
        label="Disease A",
        score=0.9,
        method_name="transformer_fake",
        rank=1,
        explanation={},
    )

    pipe = FakeExplanationPipe()
    explained = _retriever().explain_results_with_pipe(
        pipe=pipe,
        candidate_results=[candidate],
        model_name="fake-explainer",
        top_k=1,
    )

    method_specific = cast(
        dict[str, Any],
        explained[0].explanation["method_specific"],
    )

    assert pipe.prompts == ["explanation prompt"]
    assert method_specific["clinical_explanation"] == "Parsed clinical explanation."
    assert method_specific["verdict"] == "possible"
    assert method_specific["verdict_reason"] == "The disease shares seizure."
    assert method_specific["explainer_model"] == "fake-explainer"
    assert method_specific["patient_text_preview"] == "patient context"


def test_explain_results_with_pipe_handles_missing_disease_profile() -> None:
    """Test that missing disease profiles are handled without calling the pipe."""
    candidate = SimilarityResult(
        disease_id="ORPHA:MISSING",
        label="Missing Disease",
        score=0.1,
        method_name="transformer_fake",
        rank=1,
        explanation={},
    )

    explained = _retriever().explain_results_with_pipe(
        pipe=FailingPipe(),
        candidate_results=[candidate],
        model_name="fake-explainer",
        top_k=1,
    )

    method_specific = cast(
        dict[str, Any],
        explained[0].explanation["method_specific"],
    )

    assert method_specific["clinical_explanation"] == "Disease profile not found."


def test_run_stats_counts_raw_patient_terms() -> None:
    """Test LLM run statistics from patient terms and ranking count."""
    rankings = [
        SimilarityResult(
            disease_id="ORPHA:1",
            label="Disease A",
            score=1.0,
            method_name=MODEL_NAME,
            rank=1,
        )
    ]

    stats = _retriever().run_stats(rankings, elapsed=2.5)

    assert stats.n_patient_terms_raw == 1
    assert stats.n_patient_terms_propagated == 0
    assert stats.n_patient_terms_used == 1
    assert stats.n_diseases_scored == 1
    assert stats.n_diseases_skipped == 0
    assert stats.computation_time_seconds == 2.5
