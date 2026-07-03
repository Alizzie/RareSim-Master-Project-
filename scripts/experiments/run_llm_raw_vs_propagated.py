"""
Experiment: direct/raw vs propagated HPO labels for the LLM direct-retrieval
pipeline.

Reports, per mode:
- Recall@1/3/5/10/{top-k}
- MRR
- found rate
- mean rank among found cases
- flip counts: raw missed but propagated found it, and vice versa

summary are written under:
outputs/evaluation/{dataset}/experiments/

Usage:
    python -m scripts.experiments.run_llm_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json --limit 20

    python -m scripts.experiments.run_llm_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json --limit 5 --debug

    python -m scripts.experiments.run_llm_raw_vs_propagated \
        --test-set data/datasets/phenobrain_testdata/MME.json --no-cache
"""
# pylint: disable=wrong-import-position,too-few-public-methods

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raresim.core.context import AppContext
from raresim.similarity_methods.llm.config import (
    LLM_MODEL_LIST,
    MAX_NEW_TOKENS_RETRIEVAL,
    TOP_K,
)
from raresim.similarity_methods.llm.methods import (
    confidence_to_score,
    find_disease_in_profiles,
    get_hpo_label,
    load_hf_pipeline,
    query_hf,
    unload_pipeline,
)
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.io import load_json, make_safe_model_name, save_json
from raresim.utils.timer import Timer, timer

from scripts.evaluation._batch_utils import (  # pylint: disable=import-error
    EVALUATION_DIR,
    build_patient,
    load_test_cases,
)

RAW_MODE = "raw"
PROPAGATED_MODE = "propagated"
MODES = [RAW_MODE, PROPAGATED_MODE]

RECALL_KS = [1, 3, 5, 10]
DEBUG_CASE_LIMIT = 3

type GroundTruth = list[str]
type RanksByModel = dict[str, dict[str, list[int | None]]]
type TestCase = tuple[list[str], GroundTruth]

RETRIEVAL_PATTERN = re.compile(
    r"DISEASE:\s*(.+?)\s*\|\s*ORDO:\s*([\w:]+)\s*\|\s*MATCH:\s*(.+?)"
    r"\s*\|\s*CONFIDENCE:\s*([\w\s]+?)(?=DISEASE:|$)",
    re.IGNORECASE,
)


class RankingRow(TypedDict):
    """One parsed LLM disease guess with a 1-indexed rank."""

    disease_id: str
    matched_aliases: list[str]
    rank: int


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Configuration for one full raw-vs-propagated LLM experiment run."""

    test_set_path: Path
    model_name: str
    top_k: int
    limit: int | None = None
    debug: bool = False
    use_cache: bool = True


@dataclass
class ExperimentState:
    """Mutable runtime state for the experiment loop."""

    pipe: Any
    ctx: AppContext
    ancestor_sets: dict[str, set[str]]
    cache_root: Path
    ranks: RanksByModel


# ── Mode-aware term/prompt construction ──────────────────────────────────────


def get_patient_terms_for_mode(patient: PatientProfile, mode: str) -> list[str]:
    """Return direct or propagated patient terms for the experiment mode."""
    return list(patient.get_terms(use_propagated=mode == PROPAGATED_MODE))


def build_retrieval_prompt_for_mode(
    patient: PatientProfile,
    hpo_labels: dict[str, str],
    top_k: int,
    mode: str,
) -> str:
    """
    Build the LLM retrieval prompt using raw or propagated patient terms.

    Mirrors raresim.similarity_methods.llm.methods.build_retrieval_prompt
    exactly in format — the production version always uses raw terms; this
    picks terms per experiment mode so the two can be compared.
    """
    terms = get_patient_terms_for_mode(patient, mode)
    hpo_term_labels = [get_hpo_label(term, hpo_labels) for term in sorted(terms)]
    raw_text = (patient.raw_text or "").strip()

    content_parts = [
        "You are a rare disease expert specializing in clinical phenotyping.",
    ]
    if raw_text:
        content_parts.append(f"Clinical description: {raw_text}")

    content_parts += [
        f"Patient phenotypes: {', '.join(hpo_term_labels)}",
        "",
        f"You MUST list exactly {top_k} DIFFERENT rare diseases.",
        "Do not repeat the same disease.",
        f"List the top {top_k} most likely rare diseases for this patient.",
        "Format each result on a SINGLE LINE exactly like this:",
        (
            "DISEASE: <name> | ORDO: ORPHA:<number> | MATCH: <phenotypes> | "
            "CONFIDENCE: <high/medium/low>"
        ),
        "",
        "Example:",
        (
            "DISEASE: Friedreich Ataxia | ORDO: ORPHA:95 | "
            "MATCH: cerebellar ataxia, anemia | CONFIDENCE: high"
        ),
        (
            "DISEASE: Wilson Disease | ORDO: ORPHA:905 | "
            "MATCH: developmental delay, anemia | CONFIDENCE: medium"
        ),
        (
            "DISEASE: Gaucher Disease | ORDO: ORPHA:355 | "
            "MATCH: anemia, developmental delay | CONFIDENCE: low"
        ),
        "",
        f"Now list {top_k} different diseases:",
    ]

    content = "\n".join(content_parts)
    return f"[INST] {content} [/INST]"


# ── Output parsing ────────────────────────────────────────────────────────────


def clean_generated_text(generated_text: str) -> str:
    """Apply the same normalization as the production parser."""
    text = re.sub(r"ORPHA(\d+)", r"ORPHA:\1", generated_text)
    text = re.sub(r"OMIM(\d+)", r"OMIM:\1", text)
    text = re.sub(r"\[SOLUTION\]", "", text)
    text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL)
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def parse_ranked_disease_ids(
    generated_text: str,
    disease_profiles: dict[str, dict],
    top_k: int,
) -> tuple[list[RankingRow], int]:
    """
    Parse LLM output into ranked (disease_id, matched_aliases) rows.

    Lightweight version of parse_retrieval_output — skips explanation and
    category-metadata building since this experiment only needs disease
    identity and rank to compute recall/MRR, not the full output payload.

    Rows are ranked by parsed CONFIDENCE score (stable sort, so ties keep the
    LLM's original list order) — mirrors the transformer experiment's
    rank-by-score approach for an apples-to-apples methodology across both
    scripts. NOTE: this may not exactly match whatever
    raresim.core.pipeline.sort_and_rank does downstream in the production
    pipeline (not visible to this script) — if you need an exact match,
    swap the sort below for a straight parse-order ranking instead.

    Returns:
        (rows, n_validated) — n_validated counts diseases matched to a real
        profile, same diagnostic the production pipeline prints.
    """
    if not generated_text:
        return [], 0

    normalized = clean_generated_text(generated_text)

    parsed: list[tuple[str, float]] = []
    seen_ids: set[str] = set()
    n_validated = 0

    for match in RETRIEVAL_PATTERN.finditer(normalized):
        disease_name = match.group(1).strip()
        ordo_id = match.group(2).strip()
        confidence = match.group(4).strip().lower()

        matched_id, _label, validated = find_disease_in_profiles(
            ordo_id, disease_name, disease_profiles
        )

        if matched_id in seen_ids:
            continue
        seen_ids.add(matched_id)
        if validated:
            n_validated += 1

        parsed.append((matched_id, confidence_to_score(confidence)))

    parsed.sort(key=lambda row: row[1], reverse=True)

    rows: list[RankingRow] = [
        {"disease_id": disease_id, "matched_aliases": [disease_id], "rank": rank_idx}
        for rank_idx, (disease_id, _score) in enumerate(parsed[:top_k], start=1)
    ]

    return rows, n_validated


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


# ── Generation cache ──────────────────────────────────────────────────────────


def prompt_cache_path(cache_root: Path, mode: str, model_name: str, prompt: str) -> Path:
    """Return the cache file path for one exact prompt."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_dir = cache_root / mode / make_safe_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{prompt_hash}.json"


def load_cached_generation(cache_path: Path) -> str | None:
    """Return cached generated text for a prompt, or None if not cached."""
    if not cache_path.exists():
        return None
    try:
        data = load_json(cache_path)
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(data, dict) or "generated_text" not in data:
        return None
    return cast(str, data["generated_text"])


def save_cached_generation(cache_path: Path, prompt: str, generated_text: str) -> None:
    """Write generated text to the prompt cache."""
    save_json(
        {"prompt_preview": prompt[:200], "generated_text": generated_text},
        cache_path,
    )


def generate_for_mode(  # pylint: disable=too-many-arguments
    *,
    pipe: Any,
    patient: PatientProfile,
    mode: str,
    config: ExperimentRunConfig,
    hpo_labels: dict[str, str],
    cache_root: Path,
) -> tuple[str, str]:
    """
    Build the mode-specific prompt and return (prompt, generated_text).

    Uses the on-disk generation cache when enabled.
    """
    prompt = build_retrieval_prompt_for_mode(
        patient, hpo_labels, config.top_k, mode
    )

    cache_path = None
    if config.use_cache:
        cache_path = prompt_cache_path(cache_root, mode, config.model_name, prompt)
        cached = load_cached_generation(cache_path)
        if cached is not None:
            return prompt, cached

    with timer(f"generate [{mode}]"):
        generated = query_hf(prompt, pipe, max_tokens=MAX_NEW_TOKENS_RETRIEVAL)

    if cache_path is not None:
        save_cached_generation(cache_path, prompt, generated)

    return prompt, generated


# ── Sanity checks and debugging ──────────────────────────────────────────────


def debug_prompt_info(label: str, prompt: str, tokenizer: Any | None) -> None:
    """Print char/token length and a preview for one prompt."""
    token_note = ""
    if tokenizer is not None:
        try:
            n_tokens = len(tokenizer(prompt, truncation=False)["input_ids"])
            token_note = f", tokens={n_tokens}"
        except Exception as error:  # pylint: disable=broad-exception-caught
            token_note = f" (token count unavailable: {error})"

    preview = prompt[:300]
    if len(prompt) > 300:
        preview += "..."
    print(f"    {label}: chars={len(prompt)}{token_note}\n      preview: {preview}")


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
    model_summary = {
        mode: summarize_ranks(ranks[config.model_name][mode], config.top_k)
        for mode in MODES
    }
    model_summary["flips"] = summarize_flips(
        ranks[config.model_name][RAW_MODE],
        ranks[config.model_name][PROPAGATED_MODE],
    )

    return {
        "test_set": config.test_set_path.stem,
        "model_name": config.model_name,
        "top_k": config.top_k,
        "models": {config.model_name: model_summary},
    }


# ── Main experiment loop ─────────────────────────────────────────────────────


def initialize_state(config: ExperimentRunConfig) -> ExperimentState:
    """Initialize shared context, ancestor sets, and load the model ONCE."""
    out_dir = EVALUATION_DIR / config.test_set_path.stem / "experiments"
    cache_root = out_dir / "llm_raw_vs_propagated_cache"

    dummy_patient = PatientProfile("ablation_init", "", set(), set())
    ctx = AppContext.load(dummy_patient, use_canonical_profiles=True)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)

    print(f"Loading LLM pipeline: {config.model_name}")
    with timer(f"load {config.model_name}"):
        pipe = load_hf_pipeline(config.model_name, MAX_NEW_TOKENS_RETRIEVAL)

    ranks: RanksByModel = {config.model_name: {mode: [] for mode in MODES}}

    return ExperimentState(
        pipe=pipe,
        ctx=ctx,
        ancestor_sets=ancestor_sets,
        cache_root=cache_root,
        ranks=ranks,
    )


def run_case(
    index: int,
    case: TestCase,
    config: ExperimentRunConfig,
    state: ExperimentState,
) -> None:
    """Run both modes for one test case and store ranks."""
    hpo_terms, ground_truth = case
    patient = build_patient(index, hpo_terms, state.ancestor_sets)
    debug_this_case = config.debug and index < DEBUG_CASE_LIMIT
    tokenizer = getattr(state.pipe, "tokenizer", None) if debug_this_case else None

    if debug_this_case:
        print(f"\n[debug] case_{index:04d} (ground_truth={ground_truth})")

    for mode in MODES:
        prompt, generated = generate_for_mode(
            pipe=state.pipe,
            patient=patient,
            mode=mode,
            config=config,
            hpo_labels=state.ctx.hpo_labels,
            cache_root=state.cache_root,
        )

        rows, n_validated = parse_ranked_disease_ids(
            generated, state.ctx.disease_profiles, config.top_k
        )
        rank = find_first_true_rank(rows, ground_truth, state.ctx.alias_to_canonical)
        state.ranks[config.model_name][mode].append(rank)

        if debug_this_case:
            debug_prompt_info(f"prompt [{mode}]", prompt, tokenizer)
            print(
                f"    parsed [{mode}]: {len(rows)} diseases "
                f"({n_validated} validated), rank={rank}"
            )


def process_cases(
    cases: list[TestCase],
    config: ExperimentRunConfig,
    state: ExperimentState,
) -> None:
    """Run all test cases and periodically print progress."""
    total_cases = len(cases)
    for index, case in enumerate(cases):
        run_case(index, case, config, state)
        if (index + 1) % 10 == 0 or index + 1 == total_cases:
            print(f"[{index + 1:>4}/{total_cases}] processed")


def run_experiment(config: ExperimentRunConfig) -> dict[str, Any]:
    """Run the raw-vs-propagated LLM ablation over a test set."""
    cases = load_test_cases(config.test_set_path)
    if config.limit is not None:
        cases = cases[: config.limit]
    print(f"Loaded {len(cases)} test cases.\n")

    state = initialize_state(config)
    try:
        process_cases(cases, config, state)
    finally:
        unload_pipeline(state.pipe)

    return build_summary(config, state.ranks)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact raw-vs-propagated comparison table."""
    print(f"\n{'=' * 64}")
    print(
        f"  LLM Raw vs Propagated — {summary['test_set']} "
        f"({summary['model_name']}, top_k={summary['top_k']})"
    )
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
    out_path = out_dir / "llm_raw_vs_propagated.json"

    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare direct/raw vs propagated HPO labels for the LLM pipeline",
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
        "--model",
        type=str,
        default=LLM_MODEL_LIST[0],
        help=f"Model to run, default: {LLM_MODEL_LIST[0]}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N cases. LLM generation is slow — start "
            "small (e.g. 10-20) before running a full dataset."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-k results per mode, default: {TOP_K}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            f"Print prompt/token info and parsed results for the first "
            f"{DEBUG_CASE_LIMIT} cases"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the prompt->generation cache; always query the model fresh",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = ExperimentRunConfig(
        test_set_path=args.test_set,
        model_name=args.model,
        top_k=args.top_k,
        limit=args.limit,
        debug=args.debug,
        use_cache=not args.no_cache,
    )

    run_timer = Timer("llm raw_vs_propagated experiment").start()
    summary = run_experiment(config)
    run_timer.stop()

    print_summary(summary)
    out_path = save_summary(summary, args.test_set)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
