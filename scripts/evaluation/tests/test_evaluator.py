"""Tests for ``evaluator.py``.

Covers the pure, deterministic logic: ID normalisation and rank lookup,
the rank-based metrics, RRF fusion, the agreement analysis, timing
aggregation, the cache loader, and an end-to-end ``evaluate()`` run on
synthetic cases. The two output writers get smoke tests against a tmp path.
"""

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# pylint: disable=import-error
# pylint: disable=redefined-outer-name

import json
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIRS = [
    PROJECT_ROOT / "scripts" / "evaluation",
    PROJECT_ROOT / "evaluation",
]
SRC_DIR = PROJECT_ROOT / "packages" / "raresim-core" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for evaluation_dir in EVALUATION_DIRS:
    if evaluation_dir.exists() and str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))

pytest.importorskip("raresim")

import evaluator  # noqa: E402


# ── ID normalisation & rank lookup ──────────────────────────────────────────


class TestIdNormalisation:
    def test_get_disease_id_prefers_disease_id(self):
        assert evaluator.get_disease_id_from_result({"disease_id": "ORPHA:1"}) == "ORPHA:1"

    def test_get_disease_id_falls_back_to_canonical(self):
        assert (
            evaluator.get_disease_id_from_result({"canonical_disease_id": "ORPHA:2"})
            == "ORPHA:2"
        )

    def test_get_disease_id_falls_back_to_ordo(self):
        assert evaluator.get_disease_id_from_result({"ordo_id": "ORPHA:3"}) == "ORPHA:3"

    def test_get_disease_id_returns_none_when_absent(self):
        assert evaluator.get_disease_id_from_result({"score": 0.9}) is None

    def test_get_disease_id_skips_empty_string(self):
        # empty disease_id is falsy, so the fallback chain continues
        result = {"disease_id": "", "ordo_id": "ORPHA:9"}
        assert evaluator.get_disease_id_from_result(result) == "ORPHA:9"

    def test_reverse_map_is_bidirectional(self):
        reverse = evaluator.build_reverse_map({"OMIM:100": "ORPHA:1"})
        assert reverse["ORPHA:1"] == {"OMIM:100", "ORPHA:1"}
        assert reverse["OMIM:100"] == {"OMIM:100", "ORPHA:1"}

    def test_equivalent_ids_resolve_alias_both_directions(self):
        alias = {"OMIM:100": "ORPHA:1"}
        reverse = evaluator.build_reverse_map(alias)
        from_alias = evaluator.get_all_equivalent_ids("OMIM:100", alias, reverse)
        from_canonical = evaluator.get_all_equivalent_ids("ORPHA:1", alias, reverse)
        assert from_alias == {"OMIM:100", "ORPHA:1"}
        assert from_canonical == {"OMIM:100", "ORPHA:1"}

    def test_equivalent_ids_unknown_id_is_self(self):
        result = evaluator.get_all_equivalent_ids("ORPHA:999", {}, {})
        assert result == {"ORPHA:999"}


class TestFindRank:
    def test_returns_rank_of_matching_result(self):
        results = [
            {"disease_id": "ORPHA:9", "rank": 1},
            {"disease_id": "ORPHA:1", "rank": 2},
        ]
        assert evaluator.find_rank(["ORPHA:1"], results, {}, {}) == 2

    def test_returns_none_when_not_found(self):
        results = [{"disease_id": "ORPHA:9", "rank": 1}]
        assert evaluator.find_rank(["ORPHA:1"], results, {}, {}) is None

    def test_matches_via_alias(self):
        alias = {"OMIM:100": "ORPHA:1"}
        reverse = evaluator.build_reverse_map(alias)
        results = [{"disease_id": "ORPHA:1", "rank": 3}]
        assert evaluator.find_rank(["OMIM:100"], results, alias, reverse) == 3

    def test_skips_results_without_id(self):
        results = [
            {"score": 0.5, "rank": 1},  # no id -> skipped
            {"disease_id": "ORPHA:1", "rank": 2},
        ]
        assert evaluator.find_rank(["ORPHA:1"], results, {}, {}) == 2

    def test_multiple_ground_truth_ids(self):
        results = [{"disease_id": "ORPHA:2", "rank": 4}]
        assert evaluator.find_rank(["ORPHA:1", "ORPHA:2"], results, {}, {}) == 4


# ── Metrics ─────────────────────────────────────────────────────────────────


class TestNdcg:
    def test_rank_one_is_perfect(self):
        assert evaluator.compute_ndcg(1) == 1.0

    def test_rank_two(self):
        assert evaluator.compute_ndcg(2) == pytest.approx(1.0 / math.log2(3))

    def test_none_is_zero(self):
        assert evaluator.compute_ndcg(None) == 0.0

    def test_beyond_cutoff_is_zero(self):
        assert evaluator.compute_ndcg(11, top_k=10) == 0.0


class TestComputeMetrics:
    def test_empty_ranks(self):
        m = evaluator.compute_metrics([])
        assert m["found"] == 0
        assert m["median_rank"] is None
        assert m["recall_10"] == 0
        assert m["mrr"] == 0

    def test_mixed_ranks(self):
        m = evaluator.compute_metrics([1, 2, 5, None, 20])
        assert m["recall_1"] == 0.2
        assert m["recall_3"] == 0.4
        assert m["recall_5"] == 0.6
        assert m["recall_10"] == 0.6
        assert m["recall_20"] == 0.8
        assert m["mrr"] == 0.35
        assert m["found"] == 4
        assert m["median_rank"] == 5

    def test_recall_20_can_exceed_recall_10(self):
        # guards the deeper-cache fix: a rank-15 hit must lift R@20 above R@10
        m = evaluator.compute_metrics([15])
        assert m["recall_10"] == 0.0
        assert m["recall_20"] == 1.0

    def test_all_found_at_rank_one(self):
        m = evaluator.compute_metrics([1, 1, 1])
        assert m["recall_1"] == 1.0
        assert m["mrr"] == 1.0
        assert m["found"] == 3


# ── RRF ─────────────────────────────────────────────────────────────────────


class TestComputeRrf:
    def _two_method_case(self):
        return {
            "m1": [
                {"disease_id": "D1", "rank": 1},
                {"disease_id": "D2", "rank": 2},
            ],
            "m2": [
                {"disease_id": "D2", "rank": 1},
                {"disease_id": "D3", "rank": 2},
            ],
        }

    def test_disease_in_both_methods_ranks_first(self):
        fused = evaluator.compute_rrf(self._two_method_case(), ["m1", "m2"])
        ids = [r["disease_id"] for r in fused]
        # D2 is found by both methods, so it must top the fused list
        assert ids[0] == "D2"
        assert set(ids) == {"D1", "D2", "D3"}

    def test_ranks_are_contiguous_from_one(self):
        fused = evaluator.compute_rrf(self._two_method_case(), ["m1", "m2"])
        assert [r["rank"] for r in fused] == list(range(1, len(fused) + 1))

    def test_zero_weight_method_is_skipped(self):
        weights = {"m1": 2.0, "m2": 0.0}
        fused = evaluator.compute_rrf(
            self._two_method_case(), ["m1", "m2"], weights=weights
        )
        ids = {r["disease_id"] for r in fused}
        # D3 only appears in m2, which is weighted out entirely
        assert "D3" not in ids
        assert ids == {"D1", "D2"}

    def test_empty_input_yields_empty_output(self):
        assert evaluator.compute_rrf({}, ["m1"]) == []


# ── Agreement analysis ──────────────────────────────────────────────────────


class TestComputeAgreement:
    def test_counts_consensus_hard_easy_unique(self):
        results = {
            "methods": ["a", "b", "c"],
            "rank_matrix": [
                                {
                    "case_index": 0,
                    "ground_truth": ["X"],
                    "ranks": {"a": 1, "b": 1, "c": 1},
                },
                {
                    "case_index": 1,
                    "ground_truth": ["Y"],
                    "ranks": {"a": None, "b": None, "c": None},
                },
                {
                    "case_index": 2,
                    "ground_truth": ["Z"],
                    "ranks": {"a": 5, "b": None, "c": None},
                },
                {
                    "case_index": 3,
                    "ground_truth": ["W"],
                    "ranks": {"a": 2, "b": 4, "c": None},
                },
            ],
        }
        ag = evaluator.compute_agreement(results)

        assert ag["consensus_count"] == 1
        assert ag["hard_count"] == 1
        assert ag["easy_count"] == 1
        assert ag["unique_find_count"] == 1
        assert ag["hard_cases"] == ["case_0001"]
        assert ag["found_by_n"] == {3: 1, 0: 1, 1: 1, 2: 1}
        assert ag["rank_histogram"] == {1: 3, 5: 1, 2: 1, 4: 1}
        assert ag["n_methods"] == 3

    def test_unique_find_records_method_and_rank(self):
        results = {
            "methods": ["a", "b"],
            "rank_matrix": [
                {"case_index": 7, "ground_truth": ["Z"], "ranks": {"a": 5, "b": None}},
            ],
        }
        ag = evaluator.compute_agreement(results)
        assert ag["unique_finds"] == [
            {"case_id": "case_0007", "gt": ["Z"], "method": "a", "rank": 5}
        ]


# ── Timing ──────────────────────────────────────────────────────────────────


class TestTiming:
    def test_average_per_method(self):
        cases = [
            {"method_elapsed_seconds": {"a": 1.0, "b": 2.0}},
            {"method_elapsed_seconds": {"a": 3.0}},
        ]
        avg = evaluator.aggregate_method_timing(cases)
        assert avg == {"a": 2.0, "b": 2.0}

    def test_ignores_cases_without_timing(self):
        cases = [{"results": {}}, {"method_elapsed_seconds": {"a": 4.0}}]
        assert evaluator.aggregate_method_timing(cases) == {"a": 4.0}


# ── Cache loading ───────────────────────────────────────────────────────────


class TestLoadCacheDir:
    def test_missing_directory_returns_empty(self, tmp_path):
        assert evaluator.load_cache_dir(tmp_path / "nope") == []

    def test_loads_sorted_and_skips_corrupt(self, tmp_path, capsys):
        (tmp_path / "case_0000.json").write_text(json.dumps({"case_index": 0}))
        (tmp_path / "case_0001.json").write_text(json.dumps({"case_index": 1}))
        (tmp_path / "case_0002.json").write_text("{ not json")
        (tmp_path / "ignore.txt").write_text("nope")

        cases = evaluator.load_cache_dir(tmp_path)
        assert [c["case_index"] for c in cases] == [0, 1]
        assert "Could not load case_0002.json" in capsys.readouterr().out


# ── End-to-end evaluate() ───────────────────────────────────────────────────


@pytest.fixture
def synthetic_cases():
    return [
        {
            "case_index": 0,
            "hpo_terms": ["HP:0000001"],
            "ground_truth": ["ORPHA:1"],
            "results": {
                "m1": [{"disease_id": "ORPHA:1", "rank": 1}],
                "m2": [
                    {"disease_id": "ORPHA:9", "rank": 1},
                    {"disease_id": "ORPHA:1", "rank": 2},
                ],
            },
            "method_elapsed_seconds": {"m1": 0.5, "m2": 1.0},
        },
        {
            "case_index": 1,
            "hpo_terms": ["HP:0000002", "HP:0000003"],
            "ground_truth": ["ORPHA:2"],
            "results": {
                "m1": [{"disease_id": "ORPHA:7", "rank": 1}],
                "m2": [{"disease_id": "ORPHA:2", "rank": 3}],
            },
            "method_elapsed_seconds": {"m1": 0.5, "m2": 2.0},
        },
    ]


class TestEvaluate:
    def test_methods_include_three_ensembles(self, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        assert out["methods"] == [
            "m1",
            "m2",
            evaluator.RRF_METHOD_NAME,
            evaluator.RRF_WEIGHTED_NAME,
            evaluator.RRF_TOP_NAME,
        ]
        assert out["n_cases"] == 2
        assert len(out["rank_matrix"]) == 2

    def test_base_method_metrics(self, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        m1 = out["method_metrics"]["m1"]
        m2 = out["method_metrics"]["m2"]
        assert m1["recall_10"] == 0.5  # finds case 0 only
        assert m1["found"] == 1
        assert m2["recall_10"] == 1.0  # finds both
        assert m2["found"] == 2

    def test_ensemble_finds_both_cases(self, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        ens = out["method_metrics"][evaluator.RRF_METHOD_NAME]
        assert ens["found"] == 2
        assert ens["recall_10"] == 1.0

    def test_top_methods_respect_threshold(self, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        # both base methods clear R@10 >= 0.10
        assert out["rrf_top_methods"] == ["m1", "m2"]
        assert out["rrf_method_weights"] == {"m1": 0.5, "m2": 1.0}

    def test_weak_method_excluded_from_top_ensemble(self):
        # m_bad never finds the disease -> R@10 = 0 -> excluded from rrf_top
        cases = [
            {
                "case_index": i,
                "hpo_terms": [],
                "ground_truth": ["ORPHA:1"],
                "results": {
                    "m_good": [{"disease_id": "ORPHA:1", "rank": 1}],
                    "m_bad": [{"disease_id": "ORPHA:999", "rank": 1}],
                },
                "method_elapsed_seconds": {},
            }
            for i in range(3)
        ]
        out = evaluator.evaluate(cases, {}, {})
        assert out["rrf_top_methods"] == ["m_good"]

    def test_timing_averaged_into_result(self, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        assert out["method_avg_seconds"]["m1"] == 0.5
        assert out["method_avg_seconds"]["m2"] == 1.5


# ── Output writers (smoke) ──────────────────────────────────────────────────


class TestOutputWriters:
    def test_write_stats_txt(self, tmp_path, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        path = tmp_path / "stats.txt"
        evaluator.write_stats_txt(out, path)
        text = path.read_text()
        assert "top-10" in text
        assert evaluator.RRF_METHOD_NAME in text

    def test_write_summary_tsv(self, tmp_path, synthetic_cases):
        out = evaluator.evaluate(synthetic_cases, {}, {})
        path = tmp_path / "summary.tsv"
        evaluator.write_summary_tsv(out, synthetic_cases, path)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        header = lines[0].split("\t")
        assert header[:4] == ["method", "case_id", "n_hpo", "confirmed_diseases"]
        # header + (n_cases * n_methods) data rows
        assert len(lines) == 1 + 2 * len(out["methods"])
