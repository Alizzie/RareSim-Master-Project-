"""
PhenoBrain extractor — BERT-based NER via PhenoBrain public web API.

PhenoBrain uses a BERT model trained on EHR clinical notes.
The API is asynchronous: submit text → get task ID → poll for results.

No API key required.

API docs: https://github.com/xiaohaomao/timgroup_disease_diagnosis/tree/main/PhenoBrain_Web_API
"""

import time
from typing import Dict, List

from core.config import HPO_BLOCKLIST

from ._types import ExtractionMethod, ExtractionResult

# ── API settings ───────────────────────────────────────────────────────────────

_PHENOBRAIN_SUBMIT_URL  = "https://www.phenobrain.cs.tsinghua.edu.cn/extract-hpo"
_PHENOBRAIN_RESULT_URL  = "https://www.phenobrain.cs.tsinghua.edu.cn/query-extract-hpo-result"
_PHENOBRAIN_POLL_INTERVAL = 2   # seconds between polls
_PHENOBRAIN_MAX_POLLS     = 30  # give up after 60s


def extract_phenobrain_api(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    HPO extraction via PhenoBrain's public web API.

    Submits text to the API, polls until results are ready, and returns
    matched HPO terms. Async: submit → task ID → poll → parse.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        skip_negated:  Unused — PhenoBrain API handles negation internally.

    Returns:
        List of ExtractionResult for each returned HPO term.
    """
    try:
        import requests
    except ImportError:
        print("[phenobrain] requests not installed -- skipping.\n  pip install requests")
        return []

    # ── Step 1: Submit text ────────────────────────────────────────────────────
    try:
        resp = requests.post(
            _PHENOBRAIN_SUBMIT_URL,
            json={"text": raw_text, "method": "HPO/CHPO", "threshold": ""},
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("TASK_ID")
        if not task_id:
            print(f"[phenobrain] No TASK_ID in response: {resp.json()}")
            return []
    except Exception as e:
        print(f"[phenobrain] Submit failed: {e}")
        return []

    # ── Step 2: Poll for results ───────────────────────────────────────────────
    hpo_list: list = []
    hpo_to_info: dict = {}

    for _ in range(_PHENOBRAIN_MAX_POLLS):
        time.sleep(_PHENOBRAIN_POLL_INTERVAL)
        try:
            result_resp = requests.get(
                _PHENOBRAIN_RESULT_URL,
                params={"taskId": task_id},
                timeout=30,
            )
            result_resp.raise_for_status()
            data  = result_resp.json()
            state = data.get("state", "")

            if state == "SUCCESS":
                result     = data.get("result", {})
                hpo_list   = result.get("HPO_LIST", [])
                hpo_to_info = result.get("HPO_TO_INFO", {})
                break
            elif state in ("PROCESS_TEXT", "EXTRACT_HPO"):
                continue
            else:
                print(f"[phenobrain] Unexpected state: {state}")
                return []
        except Exception as e:
            print(f"[phenobrain] Poll failed: {e}")
            return []
    else:
        print("[phenobrain] Timed out waiting for results.")
        return []

    # ── Step 3: Convert to ExtractionResult ───────────────────────────────────
    results = []
    seen: set = set()

    for hpo_id in hpo_list:
        if not hpo_id or hpo_id in HPO_BLOCKLIST or hpo_id in seen:
            continue
        seen.add(hpo_id)

        info  = hpo_to_info.get(hpo_id, {})
        label = info.get("ENG_NAME") or hpo_labels.get(hpo_id, hpo_id)

        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=label,
            matched_text=hpo_id,
            method=ExtractionMethod.PHENOBRAIN_API,
            confidence=0.85,
            start=None,
            end=None,
            negated=False,
        ))

    return results
    