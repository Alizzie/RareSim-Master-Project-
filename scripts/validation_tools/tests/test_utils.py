"""
Unit tests for utils.py

Run with:
    pytest tests/test_utils.py
"""

import json
import pytest

from scripts.validation_tools._utils import (
    discover_datasets,
    resolve_datasets,
    load_cases,
    compute_stats,
    compute_mrr,
)

VALID_CASES = [
    [["HP:0001250", "HP:0002133"], ["OMIM:123456", "ORPHA:789"]],
    [["HP:0000365"], ["OMIM:654321"]],
]


@pytest.fixture
def dataset_dir(tmp_path):
    """Temporary directory with two dataset JSON files."""
    (tmp_path / "HMS.json").write_text(json.dumps(VALID_CASES))
    (tmp_path / "MME.json").write_text(json.dumps(VALID_CASES))
    return tmp_path


@pytest.fixture
def summary_found():
    """Summary where all cases are ranked."""
    return [
        {"case_id": "case_0", "rank": "1", "query_time_sec": "5.1"},
        {"case_id": "case_1", "rank": "3", "query_time_sec": "5.2"},
        {"case_id": "case_2", "rank": "10", "query_time_sec": "5.3"},
        {"case_id": "case_3", "rank": "25", "query_time_sec": "5.0"},
    ]


@pytest.fixture
def summary_mixed():
    """Summary with some unranked cases and a skipped query time."""
    return [
        {"case_id": "case_0", "rank": "1", "query_time_sec": "5.1"},
        {"case_id": "case_1", "rank": "None", "query_time_sec": None},
        {"case_id": "case_2", "rank": "5", "query_time_sec": "5.3"},
        {"case_id": "case_3", "rank": "None", "query_time_sec": ""},
    ]


# --- Discover Datasets Tests ---


class TestDiscoverDatasets:

    def test_finds_all_json_files(self, dataset_dir):
        found = discover_datasets(dataset_dir)
        assert sorted(found) == ["HMS", "MME"]

    def test_returns_empty_for_empty_dir(self, tmp_path):
        assert discover_datasets(tmp_path) == []

    def test_ignores_non_json_files(self, tmp_path):
        (tmp_path / "HMS.json").write_text(json.dumps(VALID_CASES))
        (tmp_path / "notes.txt").write_text("Ignore me.")
        found = discover_datasets(tmp_path)
        assert found == ["HMS"]

    def test_returns_sorted(self, tmp_path):
        for name in ["ZZZ", "AAA", "MMM"]:
            (tmp_path / f"{name}.json").write_text(json.dumps(VALID_CASES))
        assert discover_datasets(tmp_path) == ["AAA", "MMM", "ZZZ"]


# --- Resolve Datasets Tests ---


class TestResolveDatasets:

    def test_none_returns_all_discovered(self, dataset_dir):
        result = resolve_datasets(dataset_dir, None)
        assert sorted(result) == ["HMS", "MME"]

    def test_filters_to_requested(self, dataset_dir):
        result = resolve_datasets(dataset_dir, ["HMS"])
        assert result == ["HMS"]

    def test_case_insensitive_match(self, dataset_dir):
        result = resolve_datasets(dataset_dir, ["hms"])
        assert result == ["HMS"]

    def test_warns_and_skips_missing(self, dataset_dir, capsys):
        result = resolve_datasets(dataset_dir, ["HMS", "RAMEDIS"])
        assert result == ["HMS"]
        captured = capsys.readouterr()
        assert "RAMEDIS" in captured.out
        assert "not found" in captured.out.lower() or "skipping" in captured.out.lower()

    def test_empty_dir_returns_empty(self, tmp_path):
        result = resolve_datasets(tmp_path, None)
        assert result == []


# --- Load Cases Tests ---


class TestLoadCases:
    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text(json.dumps(VALID_CASES))
        cases = load_cases(path)
        assert len(cases) == 2
        assert cases[0] == (["HP:0001250", "HP:0002133"], ["OMIM:123456", "ORPHA:789"])

    def test_raises_on_invalid_format(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"wrong": "format"}]))
        with pytest.raises((ValueError, TypeError, KeyError)):
            load_cases(path)

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        assert load_cases(path) == []

    def test_preserves_order(self, tmp_path):
        cases = [
            [["HP:0001"], ["OMIM:111"]],
            [["HP:0002"], ["OMIM:222"]],
            [["HP:0003"], ["OMIM:333"]],
        ]
        path = tmp_path / "ordered.json"
        path.write_text(json.dumps(cases))
        loaded = load_cases(path)
        assert [c[1][0] for c in loaded] == ["OMIM:111", "OMIM:222", "OMIM:333"]


# --- Compute Stats Tests ---


class TestComputeStats:
    def test_all_found(self, summary_found):
        stats = compute_stats(summary_found, top_ks=[1, 3, 10])
        assert stats["n"] == 4
        assert stats["found"] == 4
        assert stats["not_found"] == 0
        assert stats["topk"][1] == 0.25  # only rank 1
        assert stats["topk"][3] == 0.50  # ranks 1, 3
        assert stats["topk"][10] == 0.75  # ranks 1, 3, 10

    def test_mixed_found_and_not_found(self, summary_mixed):
        stats = compute_stats(summary_mixed, top_ks=[1, 5])
        assert stats["n"] == 4
        assert stats["found"] == 2
        assert stats["not_found"] == 2
        assert stats["topk"][1] == 0.25
        assert stats["topk"][5] == 0.50

    def test_none_found(self):
        summary = [
            {"case_id": "case_0", "rank": "None", "query_time_sec": "5.0"},
            {"case_id": "case_1", "rank": "", "query_time_sec": "5.1"},
        ]
        stats = compute_stats(summary, top_ks=[1])
        assert stats["found"] == 0
        assert stats["topk"][1] == 0.0
        assert stats["median_rank"] is None

    def test_skipped_query_times_excluded_from_mean(self, summary_mixed):
        stats = compute_stats(summary_mixed)
        # only case_0 (5.1) and case_2 (5.3) have valid times
        assert stats["mean_query_time"] == pytest.approx((5.1 + 5.3) / 2, abs=0.01)

    def test_median_rank(self, summary_found):
        stats = compute_stats(summary_found)
        # ranks: 1, 3, 10, 25 → median of [1,3,10,25] = 6.5
        assert stats["median_rank"] == pytest.approx(6.5)

    def test_empty_summary(self):
        with pytest.raises((ZeroDivisionError, Exception)):
            compute_stats([])


# --- Compute MRR Tests ---


class TestComputeMrr:
    def test_all_rank_one(self):
        summary = [{"rank": "1"}, {"rank": "1"}, {"rank": "1"}]
        assert compute_mrr(summary) == pytest.approx(1.0)

    def test_mixed_ranks(self):
        # 1/1 + 1/2 + 1/4 = 1 + 0.5 + 0.25 = 1.75 / 3 ≈ 0.583
        summary = [{"rank": "1"}, {"rank": "2"}, {"rank": "4"}]
        assert compute_mrr(summary) == pytest.approx(1.75 / 3, abs=0.001)

    def test_unranked_cases_score_zero(self):
        summary = [{"rank": "1"}, {"rank": "None"}, {"rank": "None"}]
        assert compute_mrr(summary) == pytest.approx(1 / 3, abs=0.001)

    def test_empty_returns_zero(self):
        assert compute_mrr([]) == 0.0
