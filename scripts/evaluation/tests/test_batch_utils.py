"""Tests for ``_batch_utils.py``.

Covers the cache helpers that every runner depends on: path construction,
load/save round-trip behavior, merge semantics, resume checks, and result
serialization.
"""

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=wrong-import-position
# pylint: disable=import-error

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "packages" / "raresim-core" / "src"

# Support both possible project layouts.
BATCH_UTILS_DIRS = [
    PROJECT_ROOT / "scripts" / "evaluation",
    PROJECT_ROOT / "evaluation",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for batch_utils_dir in BATCH_UTILS_DIRS:
    if batch_utils_dir.exists() and str(batch_utils_dir) not in sys.path:
        sys.path.insert(0, str(batch_utils_dir))

pytest.importorskip("raresim")

import _batch_utils  # noqa: E402


# ── Test-case loading ───────────────────────────────────────────────────────


def test_load_test_cases_extracts_term_and_truth_pairs(monkeypatch):
    raw = [
        [["HP:0000001", "HP:0000002"], ["ORPHA:1"]],
        [["HP:0000003"], ["ORPHA:2", "OMIM:100"]],
    ]
    monkeypatch.setattr(_batch_utils, "load_json", lambda _path: raw)

    cases = _batch_utils.load_test_cases(Path("ignored.json"))
    assert cases == [
        (["HP:0000001", "HP:0000002"], ["ORPHA:1"]),
        (["HP:0000003"], ["ORPHA:2", "OMIM:100"]),
    ]
    assert all(isinstance(case, tuple) for case in cases)


# ── Path construction ───────────────────────────────────────────────────────


def test_cache_path_for_zero_pads_index(tmp_path):
    assert _batch_utils.cache_path_for(tmp_path, 7).name == "case_0007.json"
    assert _batch_utils.cache_path_for(tmp_path, 1234).name == "case_1234.json"


# ── load_cache ──────────────────────────────────────────────────────────────


class TestLoadCache:
    def test_missing_file_returns_skeleton(self, tmp_path):
        cache = _batch_utils.load_cache(tmp_path / "missing.json")
        assert cache == {"results": {}, "method_elapsed_seconds": {}}

    def test_reads_existing_file(self, tmp_path):
        path = tmp_path / "case_0000.json"
        payload = {"results": {"m1": []}, "method_elapsed_seconds": {"m1": 1.0}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert _batch_utils.load_cache(path) == payload

    def test_corrupt_file_returns_skeleton_with_warning(self, tmp_path, capsys):
        path = tmp_path / "case_0000.json"
        path.write_text("{ not valid json", encoding="utf-8")
        cache = _batch_utils.load_cache(path)
        assert cache == {"results": {}, "method_elapsed_seconds": {}}
        assert "Could not load cache" in capsys.readouterr().out


# ── save_cache round-trip & merge ───────────────────────────────────────────


class TestSaveCache:
    def test_basic_write_round_trip(self, tmp_path):
        path = _batch_utils.cache_path_for(tmp_path, 0)
        _batch_utils.save_cache(
            path,
            index=0,
            hpo_terms=["HP:0000002", "HP:0000001"],
            ground_truth=["ORPHA:1"],
            results={"m1": [{"disease_id": "D", "rank": 1}]},
            method_elapsed={"m1": 0.5},
            total_elapsed=0.5,
        )
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["case_index"] == 0
        assert data["hpo_terms"] == ["HP:0000001", "HP:0000002"]
        assert data["ground_truth"] == ["ORPHA:1"]
        assert data["total_elapsed_seconds"] == 0.5
        assert data["method_elapsed_seconds"] == {"m1": 0.5}
        assert data["methods_run"] == ["m1"]
        assert data["results"]["m1"] == [{"disease_id": "D", "rank": 1}]

    def test_second_method_merges_without_overwriting(self, tmp_path):
        path = _batch_utils.cache_path_for(tmp_path, 0)
        common = {
            "index": 0,
            "hpo_terms": ["HP:0000001"],
            "ground_truth": ["ORPHA:1"],
        }

        _batch_utils.save_cache(
            path,
            results={"m1": [{"disease_id": "D1", "rank": 1}]},
            method_elapsed={"m1": 0.5},
            total_elapsed=0.5,
            **common,
        )
        _batch_utils.save_cache(
            path,
            results={"m2": [{"disease_id": "D2", "rank": 1}]},
            method_elapsed={"m2": 1.5},
            total_elapsed=1.5,
            **common,
        )
        data = json.loads(path.read_text(encoding="utf-8"))

        assert set(data["results"].keys()) == {"m1", "m2"}
        assert data["methods_run"] == ["m1", "m2"]
        assert data["method_elapsed_seconds"] == {"m1": 0.5, "m2": 1.5}
        assert data["total_elapsed_seconds"] == 2.0

    def test_rerunning_same_method_overwrites_its_results(self, tmp_path):
        path = _batch_utils.cache_path_for(tmp_path, 0)
        common = {
            "index": 0,
            "hpo_terms": ["HP:0000001"],
            "ground_truth": ["ORPHA:1"],
        }

        _batch_utils.save_cache(
            path,
            results={"m1": [{"disease_id": "OLD", "rank": 1}]},
            method_elapsed={"m1": 0.5},
            total_elapsed=0.5,
            **common,
        )
        _batch_utils.save_cache(
            path,
            results={"m1": [{"disease_id": "NEW", "rank": 1}]},
            method_elapsed={"m1": 0.7},
            total_elapsed=0.7,
            **common,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["results"]["m1"] == [{"disease_id": "NEW", "rank": 1}]
        assert data["method_elapsed_seconds"]["m1"] == 0.7


# ── methods_already_cached ──────────────────────────────────────────────────


class TestMethodsAlreadyCached:
    def test_missing_file_is_not_cached(self, tmp_path):
        assert (
            _batch_utils.methods_already_cached(tmp_path / "missing.json", ["m1"])
            is False
        )

    def test_all_required_present(self, tmp_path):
        path = _batch_utils.cache_path_for(tmp_path, 0)
        _batch_utils.save_cache(
            path,
            index=0,
            hpo_terms=["HP:0000001"],
            ground_truth=["ORPHA:1"],
            results={"m1": [], "m2": []},
            method_elapsed={"m1": 0.1, "m2": 0.2},
            total_elapsed=0.3,
        )
        assert _batch_utils.methods_already_cached(path, ["m1"]) is True
        assert _batch_utils.methods_already_cached(path, ["m1", "m2"]) is True

    def test_missing_required_method(self, tmp_path):
        path = _batch_utils.cache_path_for(tmp_path, 0)
        _batch_utils.save_cache(
            path,
            index=0,
            hpo_terms=["HP:0000001"],
            ground_truth=["ORPHA:1"],
            results={"m1": []},
            method_elapsed={"m1": 0.1},
            total_elapsed=0.1,
        )
        assert _batch_utils.methods_already_cached(path, ["m1", "m3"]) is False


# ── serialize_results ───────────────────────────────────────────────────────


class _Row:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _Ranked:
    """Mimics a result container exposing a ``.rankings`` list."""

    def __init__(self, rankings):
        self.rankings = rankings


class TestSerializeResults:
    def test_object_with_rankings(self):
        results = {
            "m": _Ranked(
                [_Row({"disease_id": "D1"}), _Row({"disease_id": "D2"})]
            )
        }
        assert _batch_utils.serialize_results(results) == {
            "m": [{"disease_id": "D1"}, {"disease_id": "D2"}]
        }

    def test_list_of_mixed_objects_and_dicts(self):
        results = {"m": [_Row({"x": 1}), {"y": 2}]}
        assert _batch_utils.serialize_results(results) == {"m": [{"x": 1}, {"y": 2}]}

    def test_unrecognised_shape_becomes_empty_list(self):
        assert _batch_utils.serialize_results({"m": 123}) == {"m": []}

    def test_empty_input(self):
        assert _batch_utils.serialize_results({}) == {}
