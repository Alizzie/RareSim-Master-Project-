"""
Experiment: direct/raw vs propagated HPO labels for transformer retrieval.

Reports, per mode:
- Recall@1/3/5/10/{top-k}
- MRR
- found rate
- mean rank among found cases
- flip counts: raw missed but propagated found it, and vice versa

summary are written under:
outputs/evaluation/{dataset}/experiments/

Usage:
    python -m scripts.experiments.run_transformer_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json

    python -m scripts.experiments.run_transformer_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json --all-models

    python -m scripts.experiments.run_transformer_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json --limit 20 --debug
"""
# pylint: disable=wrong-import-position,too-few-public-methods, too-many-lines

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raresim.core.context import AppContext
from raresim.similarity_methods.transformer.config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_MODEL_LIST,
    DESCRIPTION_CHAR_BUDGET,
    MAX_LENGTH,
    MODEL_LIST,
)
from raresim.similarity_methods.transformer.methods import (
    embed_texts,
    get_model_type,
    load_embedding_backend,
)
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import get_hpo_label, preprocess_ancestor_sets
from raresim.utils.io import load_json, make_safe_model_name, save_json
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (  # pylint: disable=import-error
    EVALUATION_DIR,
    build_patient,
    load_test_cases,
)

RAW_MODE = "raw"
PROPAGATED_MODE = "propagated"
MODES = [RAW_MODE, PROPAGATED_MODE]

PROPAGATED_IC_THRESHOLD = 1.5
PROPAGATED_LABEL_CHAR_BUDGET = 650

RECALL_KS = [1, 3, 5, 10]
DEBUG_CASE_LIMIT = 3

type Array = NDArray[Any]
type DiseaseProfile = dict[str, Any]
type EmbeddingBackend = dict[str, Any]
type GroundTruth = list[str]
type RanksByModel = dict[str, dict[str, list[int | None]]]
type TestCase = tuple[list[str], GroundTruth]


class GroupedDiseaseRow(TypedDict):
    """Canonical disease row before final rank assignment."""

    disease_id: str
    representative_disease_id: str
    label: str
    score: float
    matched_aliases: list[str]


class RankingRow(GroupedDiseaseRow):
    """Canonical disease row with a 1-indexed rank."""

    rank: int


class ModelResources(TypedDict):
    """Cached resources for one transformer model."""

    model_type: str
    disease_ids: list[str]
    disease_labels: list[str]
    disease_texts: list[str]
    disease_embeddings: Array


@dataclass(frozen=True)
class TextBuildConfig:
    """Configuration for raw/propagated text construction."""

    ic_threshold: float = PROPAGATED_IC_THRESHOLD
    char_budget: int | None = PROPAGATED_LABEL_CHAR_BUDGET
    description_char_budget: int = DESCRIPTION_CHAR_BUDGET


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for one mode-specific experiment retriever."""

    model_list: list[str]
    mode: str
    cache_root: Path
    ic_values: dict[str, float] = field(default_factory=dict)
    rebuild_cache: bool = False
    text_config: TextBuildConfig = field(default_factory=TextBuildConfig)


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Configuration for one full raw-vs-propagated experiment run."""

    test_set_path: Path
    model_list: list[str]
    top_k: int
    limit: int | None = None
    debug: bool = False
    rebuild_cache: bool = False


@dataclass
class ExperimentState:
    """Mutable runtime state for the experiment loop."""

    retrievers: dict[str, "ExperimentRetriever"]
    ctx: AppContext
    ancestor_sets: dict[str, set[str]]
    ranks: RanksByModel
    tokenizer: Any | None = None


# ── Text construction ────────────────────────────────────────────────────────


def as_clean_string(value: Any) -> str:
    """Return a stripped string, treating None as empty."""
    if value is None:
        return ""
    return str(value).strip()


def as_string_list(value: Any) -> list[str]:
    """Return a value as a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def validate_mode(mode: str) -> str:
    """Validate and normalize an experiment text-construction mode."""
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"Unsupported transformer experiment mode: {mode!r}")
    return mode


def select_direct_labels(terms: list[str], hpo_labels: dict[str, str]) -> list[str]:
    """Return direct HPO labels in stable order."""
    return [get_hpo_label(term, hpo_labels) for term in sorted(terms)]


def select_propagated_labels(
    terms: list[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float] | None,
    ic_threshold: float = PROPAGATED_IC_THRESHOLD,
    char_budget: int | None = PROPAGATED_LABEL_CHAR_BUDGET,
) -> list[str]:
    """
    Select propagated phenotype labels for embedding text.

    Low-information ancestors are removed by IC threshold when IC values are
    available. Surviving terms are ordered by descending IC and capped by a
    character budget to reduce tokenizer truncation.
    """
    if not terms:
        return []

    if not ic_values:
        return [get_hpo_label(term, hpo_labels) for term in terms]

    filtered = [term for term in terms if ic_values.get(term, 0.0) >= ic_threshold]
    if not filtered:
        filtered = terms

    ranked = sorted(filtered, key=lambda term: ic_values.get(term, 0.0), reverse=True)

    if char_budget is None:
        return [get_hpo_label(term, hpo_labels) for term in ranked]

    selected: list[str] = []
    used_chars = 0
    for term in ranked:
        label = get_hpo_label(term, hpo_labels)
        added_chars = len(label) + (2 if selected else 0)
        if selected and used_chars + added_chars > char_budget:
            break
        selected.append(label)
        used_chars += added_chars

    return selected if selected else [get_hpo_label(ranked[0], hpo_labels)]


def select_labels_for_mode(
    terms: list[str],
    hpo_labels: dict[str, str],
    mode: str,
    *,
    ic_values: dict[str, float] | None = None,
    config: TextBuildConfig = TextBuildConfig(),
) -> list[str]:
    """Select labels for raw or propagated experiment mode."""
    mode = validate_mode(mode)
    if mode == PROPAGATED_MODE:
        return select_propagated_labels(
            terms,
            hpo_labels,
            ic_values,
            ic_threshold=config.ic_threshold,
            char_budget=config.char_budget,
        )
    return select_direct_labels(terms, hpo_labels)


def get_patient_terms_for_mode(patient: PatientProfile, mode: str) -> list[str]:
    """Return direct or propagated patient terms for the experiment mode."""
    return list(patient.get_terms(use_propagated=mode == PROPAGATED_MODE))


def get_disease_terms_for_mode(profile: DiseaseProfile, mode: str) -> list[str]:
    """Return direct or propagated disease terms for the experiment mode."""
    mode = validate_mode(mode)
    raw_terms = profile.get("hpo_terms")
    if mode == PROPAGATED_MODE:
        return as_string_list(profile.get("propagated_hpo_terms") or raw_terms)
    return as_string_list(raw_terms)


def build_patient_text_for_mode(
    patient: PatientProfile,
    hpo_labels: dict[str, str],
    mode: str,
    *,
    ic_values: dict[str, float] | None = None,
    config: TextBuildConfig = TextBuildConfig(),
) -> str:
    """Build patient embedding text for one experiment mode."""
    raw_text = as_clean_string(patient.raw_text)
    terms = get_patient_terms_for_mode(patient, mode)
    labels = select_labels_for_mode(
        terms,
        hpo_labels,
        mode,
        ic_values=ic_values,
        config=config,
    )

    parts: list[str] = []
    if labels:
        parts.append(f"Patient phenotypes: {'; '.join(labels)}")
    if raw_text:
        parts.append(
            f"Patient description: {raw_text[:config.description_char_budget]}"
        )

    return " ".join(parts).strip()


def build_disease_text_for_mode(
    profile: DiseaseProfile,
    hpo_labels: dict[str, str],
    mode: str,
    *,
    ic_values: dict[str, float] | None = None,
    config: TextBuildConfig = TextBuildConfig(),
) -> str:
    """Build disease embedding text for one experiment mode."""
    label = as_clean_string(profile.get("label"))
    desc = as_clean_string(profile.get("merged_description"))
    terms = get_disease_terms_for_mode(profile, mode)
    labels = select_labels_for_mode(
        terms,
        hpo_labels,
        mode,
        ic_values=ic_values,
        config=config,
    )

    parts: list[str] = []
    if label:
        parts.append(f"Disease: {label}")
    if labels:
        parts.append(f"Phenotypes: {'; '.join(labels)}")
    if desc:
        parts.append(f"Description: {desc[:config.description_char_budget]}")

    return " ".join(parts).strip()


def build_disease_texts_for_mode(
    disease_profiles: dict[str, DiseaseProfile],
    hpo_labels: dict[str, str],
    mode: str,
    *,
    ic_values: dict[str, float] | None = None,
    config: TextBuildConfig = TextBuildConfig(),
) -> tuple[list[str], list[str], list[str]]:
    """Build aligned disease IDs, labels, and texts for one experiment mode."""
    disease_ids: list[str] = []
    disease_labels: list[str] = []
    disease_texts: list[str] = []

    for disease_id, profile in disease_profiles.items():
        text = build_disease_text_for_mode(
            profile,
            hpo_labels,
            mode,
            ic_values=ic_values,
            config=config,
        )
        if not text:
            continue
        disease_ids.append(disease_id)
        disease_labels.append(as_clean_string(profile.get("label")))
        disease_texts.append(text)

    return disease_ids, disease_labels, disease_texts


# ── Ground-truth matching ────────────────────────────────────────────────────


def canonicalize(disease_id: str, alias_to_canonical: dict[str, str]) -> str:
    """Map a disease ID to its canonical form."""
    return alias_to_canonical.get(disease_id, disease_id)


def find_first_true_rank(
    rankings: list[RankingRow],
    ground_truth: GroundTruth,
    alias_to_canonical: dict[str, str],
) -> int | None:
    """Return the 1-indexed rank of the first result matching ground truth."""
    canonical_truth = {canonicalize(gt, alias_to_canonical) for gt in ground_truth}

    for result in rankings:
        candidates = {result["disease_id"], *result["matched_aliases"]}
        canonical_candidates = {
            canonicalize(candidate, alias_to_canonical) for candidate in candidates
        }
        if canonical_candidates & canonical_truth:
            return result["rank"]

    return None


# ── Sanity checks and debugging ──────────────────────────────────────────────


def check_propagated_disease_coverage(
    disease_profiles: dict[str, DiseaseProfile],
) -> None:
    """Warn if disease profiles do not really contain propagated HPO terms."""
    total = len(disease_profiles)
    if total == 0:
        return

    with_propagated = 0
    expanded = 0
    for profile in disease_profiles.values():
        raw_terms = set(as_string_list(profile.get("hpo_terms")))
        prop_terms = set(as_string_list(profile.get("propagated_hpo_terms")))
        if prop_terms:
            with_propagated += 1
        if len(prop_terms) > len(raw_terms):
            expanded += 1

    coverage = with_propagated / total
    expansion_rate = expanded / total
    print(
        f"[check] disease propagated_hpo_terms present: {coverage:.1%} "
        f"({with_propagated}/{total}); larger than raw hpo_terms: "
        f"{expansion_rate:.1%} ({expanded}/{total})"
    )
    if expansion_rate < 0.5:
        print(
            "[WARNING] Fewer than half of disease profiles show propagated "
            "terms expanding beyond raw terms. Treat propagated results as "
            "unreliable until disease-side propagation is checked."
        )


def debug_case_text(
    label: str,
    text: str,
    n_terms: int,
    tokenizer: Any | None = None,
) -> None:
    """Print term count, character length, and optional token count."""
    token_note = ""
    if tokenizer is not None:
        try:
            n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
            flag = " <-- EXCEEDS MAX_LENGTH" if n_tokens > MAX_LENGTH else ""
            token_note = f", tokens={n_tokens}{flag}"
        except Exception as error:  # pylint: disable=broad-exception-caught
            token_note = f" (token count unavailable: {error})"

    preview = text[:200]
    if len(text) > 200:
        preview += "..."

    print(
        f"    {label}: terms={n_terms}, chars={len(text)}{token_note}\n"
        f"      preview: {preview}"
    )


# ── Experiment retriever ─────────────────────────────────────────────────────


def compute_fingerprint(disease_ids: list[str], disease_texts: list[str]) -> str:
    """Hash disease IDs and mode-specific disease texts."""
    hasher = hashlib.sha256()
    for disease_id, text in zip(disease_ids, disease_texts):
        hasher.update(disease_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def build_grouped_row(
    disease_id: str,
    canonical_id: str,
    label: str,
    score: float,
) -> GroupedDiseaseRow:
    """Build the first grouped canonical row for an alias-level result."""
    return {
        "disease_id": canonical_id,
        "representative_disease_id": disease_id,
        "label": label,
        "score": score,
        "matched_aliases": [disease_id],
    }


def update_grouped_row(
    row: GroupedDiseaseRow,
    disease_id: str,
    label: str,
    score: float,
) -> None:
    """Update an existing grouped row with another alias-level result."""
    row["matched_aliases"].append(disease_id)
    if score > row["score"]:
        row["representative_disease_id"] = disease_id
        row["label"] = label
        row["score"] = score


class ExperimentRetriever:  # pylint: disable=too-many-instance-attributes
    """Minimal mode-aware retriever used only by this experiment script."""

    def __init__(
        self,
        *,
        disease_profiles: dict[str, DiseaseProfile],
        hpo_labels: dict[str, str],
        alias_to_canonical: dict[str, str],
        config: RetrieverConfig,
    ) -> None:
        self.disease_profiles = disease_profiles
        self.hpo_labels = hpo_labels
        self.alias_to_canonical = alias_to_canonical
        self.model_list = config.model_list
        self.mode = validate_mode(config.mode)
        self.cache_root = config.cache_root
        self.ic_values = config.ic_values
        self.rebuild_cache = config.rebuild_cache
        self.text_config = config.text_config

        disease_ids, disease_labels, disease_texts = build_disease_texts_for_mode(
            disease_profiles=self.disease_profiles,
            hpo_labels=self.hpo_labels,
            mode=self.mode,
            ic_values=self.ic_values,
            config=self.text_config,
        )
        self.disease_ids = disease_ids
        self.disease_labels = disease_labels
        self.disease_texts = disease_texts

        self.backends: dict[str, EmbeddingBackend] = {}
        self.registry: dict[str, ModelResources] = {}
        self.patient_embedding_cache: dict[tuple[str, str], Array] = {}

    def _cache_paths(self, model_name: str) -> dict[str, Path]:
        """Return experiment cache paths for one model and mode."""
        model_dir = self.cache_root / self.mode / make_safe_model_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ids": model_dir / "disease_ids.json",
            "labels": model_dir / "disease_labels.json",
            "texts": model_dir / "disease_texts.json",
            "embeddings": model_dir / "disease_embeddings.npy",
            "metadata": model_dir / "metadata.json",
        }

    def _metadata(self, model_name: str) -> dict[str, Any]:
        """Return expected cache metadata for the current experiment config."""
        return {
            "experiment": "transformer_raw_vs_propagated",
            "model_name": model_name,
            "model_type": get_model_type(model_name),
            "mode": self.mode,
            "max_length": MAX_LENGTH,
            "ic_threshold": self.text_config.ic_threshold,
            "propagated_char_budget": self.text_config.char_budget,
            "description_char_budget": self.text_config.description_char_budget,
            "n_diseases": len(self.disease_ids),
            "fingerprint": compute_fingerprint(self.disease_ids, self.disease_texts),
        }

    @staticmethod
    def _cache_exists(cache_paths: dict[str, Path]) -> bool:
        """Return whether all experiment cache files exist."""
        required_keys = ["ids", "labels", "texts", "embeddings", "metadata"]
        return all(cache_paths[key].exists() for key in required_keys)

    @staticmethod
    def _load_string_list(input_path: Path) -> list[str]:
        """Load a JSON list as strings."""
        data = load_json(input_path)
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {input_path}, got {type(data).__name__}")
        return [str(item) for item in data]

    def _cache_matches(
        self,
        cache_paths: dict[str, Path],
        expected: dict[str, Any],
    ) -> bool:
        """Return whether experiment cache metadata matches."""
        try:
            return load_json(cache_paths["metadata"]) == expected
        except (FileNotFoundError, ValueError, OSError):
            return False

    def _get_backend(self, model_name: str) -> EmbeddingBackend:
        """Load or return an embedding backend."""
        if model_name not in self.backends:
            self.backends[model_name] = load_embedding_backend(model_name)
        return self.backends[model_name]

    def get_tokenizer(self, model_name: str) -> Any | None:
        """Best-effort tokenizer access for debug token-length checks."""
        backend = self._get_backend(model_name)
        if backend["model_type"] == "hf_encoder":
            return backend["tokenizer"]
        if backend["model_type"] == "sentence_transformer":
            return getattr(backend["model"], "tokenizer", None)
        return None

    def warmup(self) -> None:
        """Ensure disease embeddings exist for all selected models."""
        for model_name in self.model_list:
            print(f"  Preparing [{self.mode}]: {model_name}")
            self._ensure_model_resources(model_name)

    def _has_valid_cache(
        self,
        cache_paths: dict[str, Path],
        expected_metadata: dict[str, Any],
    ) -> bool:
        """Return whether a reusable cache exists for one model."""
        return (
            self._cache_exists(cache_paths)
            and not self.rebuild_cache
            and self._cache_matches(cache_paths, expected_metadata)
        )

    def _load_model_resources(self, cache_paths: dict[str, Path]) -> ModelResources:
        """Load model resources from an existing experiment cache."""
        return {
            "model_type": "",
            "disease_ids": self._load_string_list(cache_paths["ids"]),
            "disease_labels": self._load_string_list(cache_paths["labels"]),
            "disease_texts": self._load_string_list(cache_paths["texts"]),
            "disease_embeddings": np.load(cache_paths["embeddings"]),
        }

    def _build_model_resources(self, model_name: str) -> ModelResources:
        """Build model resources by embedding the mode-specific disease texts."""
        backend = self._get_backend(model_name)
        return {
            "model_type": get_model_type(model_name),
            "disease_ids": self.disease_ids,
            "disease_labels": self.disease_labels,
            "disease_texts": self.disease_texts,
            "disease_embeddings": embed_texts(backend, self.disease_texts),
        }

    @staticmethod
    def _save_model_resources(
        resources: ModelResources,
        cache_paths: dict[str, Path],
        metadata: dict[str, Any],
    ) -> None:
        """Save model resources to the experiment cache."""
        save_json(resources["disease_ids"], cache_paths["ids"])
        save_json(resources["disease_labels"], cache_paths["labels"])
        save_json(resources["disease_texts"], cache_paths["texts"])
        np.save(cache_paths["embeddings"], resources["disease_embeddings"])
        save_json(metadata, cache_paths["metadata"])

    def _ensure_model_resources(self, model_name: str) -> None:
        """Build or load disease embeddings for one model."""
        if model_name in self.registry:
            return

        cache_paths = self._cache_paths(model_name)
        expected_metadata = self._metadata(model_name)

        if self._has_valid_cache(cache_paths, expected_metadata):
            print(f"    Loading experiment cache [{self.mode}]: {model_name}")
            resources = self._load_model_resources(cache_paths)
            resources["model_type"] = get_model_type(model_name)
        else:
            if self._cache_exists(cache_paths) and not self.rebuild_cache:
                print(
                    f"    Experiment cache mismatch [{self.mode}], rebuilding: "
                    f"{model_name}"
                )
            else:
                print(f"    Building experiment cache [{self.mode}]: {model_name}")
            resources = self._build_model_resources(model_name)
            self._save_model_resources(resources, cache_paths, expected_metadata)

        self.registry[model_name] = resources

    def _get_patient_embedding(self, model_name: str, patient_text: str) -> Array:
        """Embed patient text with in-memory caching."""
        text_hash = hashlib.sha256(patient_text.encode("utf-8")).hexdigest()
        cache_key = (model_name, text_hash)
        if cache_key in self.patient_embedding_cache:
            return self.patient_embedding_cache[cache_key]

        backend = self._get_backend(model_name)
        embedding = embed_texts(backend, [patient_text])[0]
        self.patient_embedding_cache[cache_key] = embedding
        return embedding

    def rank(
        self,
        *,
        model_name: str,
        patient: PatientProfile,
        top_k: int,
        candidate_pool_size: int,
    ) -> list[RankingRow]:
        """Return canonicalized rankings for one patient/model/mode."""
        self._ensure_model_resources(model_name)

        patient_text = build_patient_text_for_mode(
            patient,
            self.hpo_labels,
            self.mode,
            ic_values=self.ic_values,
            config=self.text_config,
        )
        if not patient_text:
            raise ValueError("Patient text is empty.")

        patient_embedding = self._get_patient_embedding(model_name, patient_text)
        resources = self.registry[model_name]
        scores = resources["disease_embeddings"] @ patient_embedding
        pool_size = min(candidate_pool_size, len(scores))
        ranked_indices = np.argsort(-scores)[:pool_size]

        return self._collapse_to_canonical(ranked_indices, scores, resources, top_k)

    def _collapse_to_canonical(  # pylint: disable=too-many-locals
        self,
        ranked_indices: Array,
        scores: Array,
        resources: ModelResources,
        top_k: int,
    ) -> list[RankingRow]:
        """Collapse alias-level ranked rows into canonical disease rows."""
        grouped: dict[str, GroupedDiseaseRow] = {}
        disease_ids = resources["disease_ids"]
        disease_labels = resources["disease_labels"]

        for idx in ranked_indices:
            position = int(idx)
            disease_id = disease_ids[position]
            canonical_id = canonicalize(disease_id, self.alias_to_canonical)
            label = disease_labels[position]
            score = float(scores[position])

            if canonical_id in grouped:
                update_grouped_row(grouped[canonical_id], disease_id, label, score)
            else:
                grouped[canonical_id] = build_grouped_row(
                    disease_id,
                    canonical_id,
                    label,
                    score,
                )

        collapsed = sorted(grouped.values(), key=lambda row: row["score"], reverse=True)
        results: list[RankingRow] = []
        for rank_idx, row in enumerate(collapsed[:top_k], start=1):
            results.append(cast(RankingRow, {**row, "rank": rank_idx}))
        return results


# ── Resources ────────────────────────────────────────────────────────────────


def load_retrievers(
    *,
    model_list: list[str],
    cache_root: Path,
    rebuild_cache: bool = False,
) -> tuple[dict[str, ExperimentRetriever], AppContext]:
    """Build one experiment retriever per mode."""
    dummy_patient = PatientProfile("ablation_init", "", set(), set())
    ctx = AppContext.load(dummy_patient, use_canonical_profiles=True)

    check_propagated_disease_coverage(ctx.disease_profiles)

    retrievers: dict[str, ExperimentRetriever] = {}
    for mode in MODES:
        print(
            f"Preparing experiment retriever [{mode}]"
            f"{' (rebuilding cache)' if rebuild_cache else ''}..."
        )
        config = RetrieverConfig(
            model_list=model_list,
            mode=mode,
            cache_root=cache_root,
            ic_values=ctx.ic_values,
            rebuild_cache=rebuild_cache,
        )
        retriever = ExperimentRetriever(
            disease_profiles=ctx.disease_profiles,
            hpo_labels=ctx.hpo_labels,
            alias_to_canonical=ctx.alias_to_canonical,
            config=config,
        )
        retriever.warmup()
        retrievers[mode] = retriever

    return retrievers, ctx


def initialize_state(config: ExperimentRunConfig) -> ExperimentState:
    """Initialize retrievers, shared context, and rank storage."""
    out_dir = EVALUATION_DIR / config.test_set_path.stem / "experiments"
    cache_root = out_dir / "transformer_raw_vs_propagated_cache"
    retrievers, ctx = load_retrievers(
        model_list=config.model_list,
        cache_root=cache_root,
        rebuild_cache=config.rebuild_cache,
    )
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    ranks: RanksByModel = {
        model_name: {mode: [] for mode in MODES} for model_name in config.model_list
    }

    tokenizer = None
    if config.debug and config.model_list:
        tokenizer = retrievers[RAW_MODE].get_tokenizer(config.model_list[0])

    return ExperimentState(
        retrievers=retrievers,
        ctx=ctx,
        ancestor_sets=ancestor_sets,
        ranks=ranks,
        tokenizer=tokenizer,
    )


# ── Metrics ──────────────────────────────────────────────────────────────────


def summarize_ranks(ranks: list[int | None], top_k: int) -> dict[str, Any]:
    """Compute Recall@k, MRR, found rate, and mean rank among found cases."""
    n_cases = len(ranks)
    found = [rank for rank in ranks if rank is not None]

    recall_at: dict[str, float] = {}
    for cutoff in [*RECALL_KS, top_k]:
        if cutoff > top_k:
            continue
        recall_at[f"recall@{cutoff}"] = (
            round(sum(1 for rank in found if rank <= cutoff) / n_cases, 4)
            if n_cases
            else 0.0
        )

    mrr = round(sum(1.0 / rank for rank in found) / n_cases, 4) if n_cases else 0.0
    mean_rank_found = round(sum(found) / len(found), 2) if found else None

    return {
        "n_cases": n_cases,
        "n_found": len(found),
        "found_rate": round(len(found) / n_cases, 4) if n_cases else 0.0,
        **recall_at,
        "mrr": mrr,
        "mean_rank_found": mean_rank_found,
    }


def summarize_flips(
    raw_ranks: list[int | None],
    propagated_ranks: list[int | None],
) -> dict[str, int]:
    """Count cases where only one mode found the ground-truth disease."""
    raw_miss_propagated_hit = sum(
        1
        for raw_rank, propagated_rank in zip(raw_ranks, propagated_ranks)
        if raw_rank is None and propagated_rank is not None
    )
    propagated_miss_raw_hit = sum(
        1
        for raw_rank, propagated_rank in zip(raw_ranks, propagated_ranks)
        if raw_rank is not None and propagated_rank is None
    )
    return {
        "raw_miss_propagated_hit": raw_miss_propagated_hit,
        "propagated_miss_raw_hit": propagated_miss_raw_hit,
    }


def build_summary(config: ExperimentRunConfig, ranks: RanksByModel) -> dict[str, Any]:
    """Build the final JSON-serializable experiment summary."""
    summary: dict[str, Any] = {
        "test_set": config.test_set_path.stem,
        "top_k": config.top_k,
        "models": {},
    }

    for model_name in config.model_list:
        model_summary = {
            mode: summarize_ranks(ranks[model_name][mode], config.top_k)
            for mode in MODES
        }
        model_summary["flips"] = summarize_flips(
            ranks[model_name][RAW_MODE],
            ranks[model_name][PROPAGATED_MODE],
        )
        summary["models"][model_name] = model_summary

    return summary


# ── Main experiment loop ─────────────────────────────────────────────────────


def print_debug_case(
    *,
    index: int,
    ground_truth: GroundTruth,
    patient: PatientProfile,
    retrievers: dict[str, ExperimentRetriever],
    tokenizer: Any | None = None,
) -> None:
    """Print text diagnostics for one case."""
    print(f"\n[debug] case_{index:04d} (ground_truth={ground_truth})")
    for mode in MODES:
        retriever = retrievers[mode]
        text = build_patient_text_for_mode(
            patient,
            retriever.hpo_labels,
            mode,
            ic_values=retriever.ic_values,
            config=retriever.text_config,
        )
        terms = get_patient_terms_for_mode(patient, mode)
        labels = select_labels_for_mode(
            terms,
            retriever.hpo_labels,
            mode,
            ic_values=retriever.ic_values,
            config=retriever.text_config,
        )
        if mode == PROPAGATED_MODE and len(labels) != len(terms):
            print(
                f"    [{mode}] IC filter+budget: {len(terms)} -> "
                f"{len(labels)} labels"
            )
        debug_case_text(f"patient [{mode}]", text, len(labels), tokenizer)


def run_case(
    index: int,
    case: TestCase,
    config: ExperimentRunConfig,
    state: ExperimentState,
) -> None:
    """Run all selected models/modes for one test case and store ranks."""
    hpo_terms, ground_truth = case
    patient = build_patient(index, hpo_terms, state.ancestor_sets)

    if config.debug and index < DEBUG_CASE_LIMIT:
        print_debug_case(
            index=index,
            ground_truth=ground_truth,
            patient=patient,
            retrievers=state.retrievers,
            tokenizer=state.tokenizer,
        )

    for mode in MODES:
        retriever = state.retrievers[mode]
        for model_name in config.model_list:
            rankings = retriever.rank(
                model_name=model_name,
                patient=patient,
                top_k=config.top_k,
                candidate_pool_size=CANDIDATE_POOL_SIZE,
            )
            rank = find_first_true_rank(
                rankings,
                ground_truth,
                state.ctx.alias_to_canonical,
            )
            state.ranks[model_name][mode].append(rank)


def process_cases(
    cases: list[TestCase],
    config: ExperimentRunConfig,
    state: ExperimentState,
) -> None:
    """Run all test cases and periodically print progress."""
    total_cases = len(cases)
    for index, case in enumerate(cases):
        run_case(index, case, config, state)
        if (index + 1) % 25 == 0 or index + 1 == total_cases:
            print(f"[{index + 1:>4}/{total_cases}] processed")


def run_experiment(config: ExperimentRunConfig) -> dict[str, Any]:
    """Run the raw-vs-propagated ablation over a test set."""
    cases = load_test_cases(config.test_set_path)
    if config.limit is not None:
        cases = cases[: config.limit]
    print(f"Loaded {len(cases)} test cases.\n")

    state = initialize_state(config)
    process_cases(cases, config, state)
    return build_summary(config, state.ranks)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact raw-vs-propagated comparison table."""
    print(f"\n{'=' * 64}")
    print(f"  Raw vs Propagated — {summary['test_set']} (top_k={summary['top_k']})")
    print("=" * 64)

    for model_name, model_summary in summary["models"].items():
        print(f"\n{model_name}")
        for mode in MODES:
            stats = model_summary[mode]
            metrics = ", ".join(
                f"{key}={value}"
                for key, value in stats.items()
                if key.startswith("recall@")
            )
            print(
                f"  {mode:<11} found={stats['found_rate']:.2%} "
                f"mrr={stats['mrr']} {metrics}"
            )
        flips = model_summary["flips"]
        print(
            "  flips        raw miss -> propagated hit: "
            f"{flips['raw_miss_propagated_hit']} | "
            "propagated miss -> raw hit: "
            f"{flips['propagated_miss_raw_hit']}"
        )

    print(f"\n{'=' * 64}\n")


def save_summary(summary: dict[str, Any], test_set_path: Path) -> Path:
    """Write summary JSON under outputs/evaluation/{dataset}/experiments/."""
    out_dir = EVALUATION_DIR / test_set_path.stem / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transformer_raw_vs_propagated.json"

    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare direct/raw vs propagated HPO labels for transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to test set JSON, e.g. test_data/test_cases/MME.json",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all transformer models instead of just the default one",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N cases, useful for a quick check",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k results per method, default: 20",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            f"Print term counts, text previews, and token counts for the first "
            f"{DEBUG_CASE_LIMIT} cases"
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force fresh experiment disease-embedding caches",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    model_list = MODEL_LIST if args.all_models else DEFAULT_MODEL_LIST
    config = ExperimentRunConfig(
        test_set_path=args.test_set,
        model_list=model_list,
        top_k=args.top_k,
        limit=args.limit,
        debug=args.debug,
        rebuild_cache=args.rebuild_cache,
    )

    timer = Timer("raw_vs_propagated experiment").start()
    summary = run_experiment(config)
    timer.stop()

    print_summary(summary)
    out_path = save_summary(summary, args.test_set)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
