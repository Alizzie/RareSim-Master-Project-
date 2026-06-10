"""
Small focused tests for runner-specific logic.
API calls are mocked — no running services required.

Run with:
    pytest tests/test_runners.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

# ── LIRICAL ───────────────────────────────────────────────────────────────────

from run_lirical import parse_lirical_tsv, find_best_rank


class TestParseLiricalTsv:
    def test_parses_valid_tsv(self, tmp_path):
        tsv = tmp_path / "case.tsv"
        tsv.write_text(
            "rank\tdiseaseCurie\tdiseaseName\tposttestprob\n"
            "1\tOMIM:123456\tSome Disease\t0.95\n"
            "2\tORPHA:789\tOther Disease\t0.80\n"
        )
        rows = parse_lirical_tsv(tsv)
        assert len(rows) == 2
        assert rows[0]["rank"] == 1
        assert rows[0]["disease_id"] == "OMIM:123456"
        assert rows[0]["post_test_prob"] == "0.95"

    def test_skips_comment_lines(self, tmp_path):
        tsv = tmp_path / "case.tsv"
        tsv.write_text(
            "! This is a comment\n"
            "rank\tdiseaseCurie\tdiseaseName\tposttestprob\n"
            "1\tOMIM:123456\tSome Disease\t0.95\n"
        )
        rows = parse_lirical_tsv(tsv)
        assert len(rows) == 1

    def test_returns_empty_for_missing_file(self, tmp_path):
        rows = parse_lirical_tsv(tmp_path / "nonexistent.tsv")
        assert rows == []


class TestFindBestRankLirical:
    def test_finds_exact_match(self):
        results = [
            {"rank": 1, "disease_id": "OMIM:999", "post_test_prob": "0.9"},
            {"rank": 2, "disease_id": "OMIM:123456", "post_test_prob": "0.8"},
        ]
        rank, matched, score = find_best_rank(results, ["OMIM:123456"])
        assert rank == 2
        assert matched == "OMIM:123456"

    def test_case_insensitive_match(self):
        results = [{"rank": 1, "disease_id": "omim:123456", "post_test_prob": "0.9"}]
        rank, matched, score = find_best_rank(results, ["OMIM:123456"])
        assert rank == 1

    def test_picks_lowest_rank_among_multiple_confirmed(self):
        results = [
            {"rank": 3, "disease_id": "OMIM:111", "post_test_prob": "0.7"},
            {"rank": 1, "disease_id": "OMIM:222", "post_test_prob": "0.9"},
        ]
        rank, matched, score = find_best_rank(results, ["OMIM:111", "OMIM:222"])
        assert rank == 1
        assert matched == "OMIM:222"

    def test_returns_none_when_not_found(self):
        results = [{"rank": 1, "disease_id": "OMIM:999", "post_test_prob": "0.9"}]
        rank, matched, score = find_best_rank(results, ["OMIM:123456"])
        assert rank is None
        assert matched is None

    def test_empty_results(self):
        rank, matched, score = find_best_rank([], ["OMIM:123456"])
        assert rank is None


# ── PhenoBrain ────────────────────────────────────────────────────────────────

from run_phenobrain import get_disease_ranking, create_RD_code_mapper


class TestGetDiseaseRankingPhenobrain:
    def test_basic_ranking(self):
        results = [
            {"CODE": "RD:100", "SCORE": 0.99},
            {"CODE": "RD:200", "SCORE": 0.95},
        ]
        mapper = {"RD:100": "OMIM:111", "RD:200": "OMIM:222"}
        ranking = get_disease_ranking(results, mapper)
        assert ranking[0] == ("OMIM:111", 0.99)
        assert ranking[1] == ("OMIM:222", 0.95)

    def test_unmapped_code_returns_none(self):
        results = [{"CODE": "RD:999", "SCORE": 0.9}]
        ranking = get_disease_ranking(results, {})
        assert ranking[0] == (None, 0.9)

    def test_empty_results(self):
        assert get_disease_ranking([], {}) == {}


class TestPhenobrainApiConnectivity:
    """Smoke test — checks the predict endpoint is reachable and returns a task ID."""

    def test_predict_returns_task_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"TASK_ID": "abc-123"}

        with patch("run_phenobrain.requests.get", return_value=mock_response):
            from run_phenobrain import predict_case

            status, task_id = predict_case(["HP:0001250"], topk=5)

        assert status is True
        assert task_id == "abc-123"

    def test_predict_handles_api_failure(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        with patch("run_phenobrain.requests.get", return_value=mock_response):
            from run_phenobrain import predict_case

            status, task_id = predict_case(["HP:0001250"], topk=5)

        assert status is False
        assert task_id is None


# ── DX29 Search ───────────────────────────────────────────────────────────────

from run_dx29_search import (
    predict_case as dx29_predict,
    get_disease_ranking as dx29_ranking,
)


class TestDx29SearchApiConnectivity:
    def test_returns_disease_list_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "diseases": [
                {"id": "ORPHA:123", "score": 0.95},
                {"id": "ORPHA:456", "score": 0.80},
            ]
        }

        with patch("run_dx29_search.requests.post", return_value=mock_response):
            status, results = dx29_predict(
                ["HP:0001250"], host="http://localhost:8080", lang="en", count=10
            )

        assert status is True
        assert len(results) == 2
        assert results[0]["id"] == "ORPHA:123"

    def test_handles_connection_error(self):
        import requests

        with patch(
            "run_dx29_search.requests.post",
            side_effect=requests.exceptions.ConnectionError,
        ):
            with pytest.raises(requests.exceptions.ConnectionError):
                dx29_predict(
                    ["HP:0001250"], host="http://localhost:8080", lang="en", count=10
                )

    def test_handles_non_200_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 503
        with patch("run_dx29_search.requests.post", return_value=mock_response):
            status, results = dx29_predict(
                ["HP:0001250"], host="http://localhost:8080", lang="en", count=10
            )
        assert status is False
        assert results is None


class TestDx29GetDiseaseRanking:
    def test_basic_ranking(self):
        results = [
            {"id": "ORPHA:123", "score": 0.95},
            {"id": "ORPHA:456", "score": 0.80},
        ]
        ranking = dx29_ranking(results)
        assert ranking[0] == ("ORPHA:123", 0.95)
        assert ranking[1] == ("ORPHA:456", 0.80)

    def test_empty_results(self):
        assert dx29_ranking([]) == {}


# ── DX29 Phrank ───────────────────────────────────────────────────────────────

from run_dx29_phrank import predict_case as phrank_predict


class TestDx29PhrankApiConnectivity:
    def test_returns_ranked_list_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "ORPHA:123", "score": 0.95},
            {"id": "ORPHA:456", "score": 0.80},
        ]

        with patch("run_dx29_phrank.requests.post", return_value=mock_response):
            status, results = phrank_predict(
                ["HP:0001250"], topk=10, host="http://localhost:8080", lang="en"
            )

        assert status is True
        assert len(results) == 2

    def test_handles_non_200_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        with patch("run_dx29_phrank.requests.post", return_value=mock_response):
            status, results = phrank_predict(
                ["HP:0001250"], topk=10, host="http://localhost:8080", lang="en"
            )
        assert status is False
        assert results is None
