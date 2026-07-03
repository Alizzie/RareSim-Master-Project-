"""
PhenoBrain extractor — BERT-based NER via PhenoBrain public web API.

PhenoBrain uses a BERT model trained on EHR clinical notes.
The API is asynchronous: submit text, get task ID, then poll for results.

No API key required.

API docs:
https://github.com/xiaohaomao/timgroup_disease_diagnosis/tree/main/PhenoBrain_Web_API
"""

import importlib
import time
from typing import Any

from raresim.hpo_extraction._config import HPO_BLOCKLIST

from ._types import ExtractionMethod, ExtractionResult

# ── Optional requests dependency ───────────────────────────────────────────────

_requests_module: Any | None = None
RequestException: type[BaseException] = RuntimeError

try:
    _requests_module = importlib.import_module("requests")
except ImportError:
    pass
else:
    _exceptions_module = getattr(_requests_module, "exceptions", None)
    _request_exception = getattr(_exceptions_module, "RequestException", RuntimeError)

    if isinstance(_request_exception, type) and issubclass(
        _request_exception, BaseException
    ):
        RequestException = _request_exception


# ── API settings ───────────────────────────────────────────────────────────────

_PHENOBRAIN_SUBMIT_URL = "https://www.phenobrain.cs.tsinghua.edu.cn/extract-hpo"
_PHENOBRAIN_RESULT_URL = (
    "https://www.phenobrain.cs.tsinghua.edu.cn/query-extract-hpo-result"
)
_PHENOBRAIN_POLL_INTERVAL = 2
_PHENOBRAIN_MAX_POLLS = 30
_PHENOBRAIN_PENDING_STATES = {"PROCESS_TEXT", "EXTRACT_HPO"}


def _get_requests_module() -> Any | None:
    """Return the optional requests module if installed."""
    if _requests_module is None:
        print("[phenobrain] requests not installed -- skipping.\n  pip install requests")
        return None

    return _requests_module


def _submit_text(raw_text: str) -> str | None:
    """Submit clinical text to PhenoBrain and return the task ID."""
    requests_module = _get_requests_module()
    if requests_module is None:
        return None

    try:
        response = requests_module.post(
            _PHENOBRAIN_SUBMIT_URL,
            json={"text": raw_text, "method": "HPO/CHPO", "threshold": ""},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except RequestException as exc:
        print(f"[phenobrain] Submit failed: {exc}")
        return None
    except ValueError as exc:
        print(f"[phenobrain] Invalid submit response JSON: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[phenobrain] Unexpected submit response: {data}")
        return None

    task_id = data.get("TASK_ID")
    if not isinstance(task_id, str) or not task_id:
        print(f"[phenobrain] No TASK_ID in response: {data}")
        return None

    return task_id


def _poll_result(
    task_id: str,
) -> tuple[list[str], dict[str, dict[str, Any]]] | None:
    """Poll PhenoBrain until extraction results are available."""
    requests_module = _get_requests_module()
    if requests_module is None:
        return None

    for _ in range(_PHENOBRAIN_MAX_POLLS):
        time.sleep(_PHENOBRAIN_POLL_INTERVAL)

        data = _fetch_poll_data(requests_module, task_id)
        if data is None:
            return None

        state = data.get("state", "")
        if state == "SUCCESS":
            return _parse_result_payload(data)

        if state in _PHENOBRAIN_PENDING_STATES:
            continue

        print(f"[phenobrain] Unexpected state: {state}")
        return None

    print("[phenobrain] Timed out waiting for results.")
    return None


def _fetch_poll_data(requests_module: Any, task_id: str) -> dict[str, Any] | None:
    """Fetch one PhenoBrain poll response."""
    try:
        response = requests_module.get(
            _PHENOBRAIN_RESULT_URL,
            params={"taskId": task_id},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except RequestException as exc:
        print(f"[phenobrain] Poll failed: {exc}")
        return None
    except ValueError as exc:
        print(f"[phenobrain] Invalid poll response JSON: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[phenobrain] Unexpected poll response: {data}")
        return None

    return data


def _parse_result_payload(
    data: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Parse HPO IDs and metadata from a successful PhenoBrain response."""
    result = data.get("result", {})
    if not isinstance(result, dict):
        return [], {}

    hpo_list = _parse_hpo_list(result.get("HPO_LIST", []))
    hpo_to_info = _parse_hpo_to_info(result.get("HPO_TO_INFO", {}))

    return hpo_list, hpo_to_info


def _parse_hpo_list(value: Any) -> list[str]:
    """Parse a PhenoBrain HPO_LIST value."""
    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, str) and item]


def _parse_hpo_to_info(value: Any) -> dict[str, dict[str, Any]]:
    """Parse a PhenoBrain HPO_TO_INFO value."""
    if not isinstance(value, dict):
        return {}

    parsed: dict[str, dict[str, Any]] = {}

    for hpo_id, info in value.items():
        if isinstance(hpo_id, str) and isinstance(info, dict):
            parsed[hpo_id] = info

    return parsed


def _get_label(
    hpo_id: str,
    hpo_to_info: dict[str, dict[str, Any]],
    hpo_labels: dict[str, str],
) -> str:
    """Return the best available label for one HPO ID."""
    info = hpo_to_info.get(hpo_id, {})
    api_label = info.get("ENG_NAME")

    if isinstance(api_label, str) and api_label:
        return api_label

    return hpo_labels.get(hpo_id) or hpo_id


def _build_extraction_result(
    hpo_id: str,
    label: str,
) -> ExtractionResult:
    """Build one PhenoBrain extraction result."""
    return ExtractionResult(
        hpo_id=hpo_id,
        label=label,
        matched_text=hpo_id,
        method=ExtractionMethod.PHENOBRAIN_API,
        confidence=0.85,
        start=None,
        end=None,
        negated=False,
    )


def _convert_results(
    hpo_list: list[str],
    hpo_to_info: dict[str, dict[str, Any]],
    hpo_labels: dict[str, str],
) -> list[ExtractionResult]:
    """Convert PhenoBrain HPO IDs into extraction results."""
    results: list[ExtractionResult] = []
    seen: set[str] = set()

    for hpo_id in hpo_list:
        if hpo_id in HPO_BLOCKLIST or hpo_id in seen:
            continue

        seen.add(hpo_id)
        label = _get_label(hpo_id, hpo_to_info, hpo_labels)
        results.append(_build_extraction_result(hpo_id, label))

    return results


def extract_phenobrain_api(
    raw_text: str,
    hpo_labels: dict[str, str],
    skip_negated: bool = True,
) -> list[ExtractionResult]:
    """
    HPO extraction via PhenoBrain's public web API.

    Submits text to the API, polls until results are ready, and returns
    matched HPO terms.

    Args:
        raw_text: Raw clinical patient text.
        hpo_labels: Dict mapping HPO ID to label string.
        skip_negated: Accepted for extractor interface compatibility.

    Returns:
        Extraction results for each returned HPO term.
    """
    _ = skip_negated

    task_id = _submit_text(raw_text)
    if task_id is None:
        return []

    poll_result = _poll_result(task_id)
    if poll_result is None:
        return []

    hpo_list, hpo_to_info = poll_result
    return _convert_results(hpo_list, hpo_to_info, hpo_labels)
