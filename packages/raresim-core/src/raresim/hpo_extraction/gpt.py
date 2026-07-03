"""
GPT extractor — HPO extraction via GPT-4o-mini.

The model extracts phenotype phrases from clinical text. Phrases are then
mapped locally to HPO IDs via hpo_labels. The model never generates ontology
identifiers directly, reducing hallucination risk.

Requires:
    pip install openai
    OPENAI_API_KEY=sk-... in .env file

Paper: https://ieeexplore.ieee.org/document/10340611
"""

import importlib
import json
import os
import re
from typing import Any

from raresim.hpo_extraction._config import HPO_BLOCKLIST

from ._types import ExtractionMethod, ExtractionResult
from ._utils import build_label_lookup, normalize_text

# ── Optional OpenAI dependency ─────────────────────────────────────────────────

_openai_client_class: Any | None = None

try:
    _openai_module = importlib.import_module("openai")
except ImportError:
    pass
else:
    _openai_client_class = getattr(_openai_module, "OpenAI", None)


# ── Model settings ─────────────────────────────────────────────────────────────

_CHATGPT_MODEL = "gpt-4o-mini"

_CHATGPT_SYSTEM_PROMPT = """You are a clinical phenotype extraction expert.

Given clinical text, extract only explicitly mentioned abnormal human phenotype phrases.

Rules:
- Return ONLY valid JSON.
- Do not return markdown.
- Do not return explanations.
- Do not infer unstated phenotypes.
- Do not return HPO IDs.
- Do not include diagnoses, disease names, genes, treatments, inheritance patterns, or normal findings.
- Extract at most 20 phenotype phrases.
- Use short canonical medical phrases such as "microcephaly", "global developmental delay", or "hypotonia".

Output format:
{"phenotypes": ["microcephaly", "global developmental delay", "hypotonia"]}

If no phenotype is found, return:
{"phenotypes": []}
"""


def _build_user_prompt(raw_text: str, skip_negated: bool) -> str:
    """Build the user prompt sent to the GPT extractor."""
    negation_rule = ""

    if skip_negated:
        negation_rule = (
            "Do not extract findings described as absent, denied, ruled out, "
            "negative, normal, or not present.\n\n"
        )

    return f"{negation_rule}Clinical text:\n{raw_text}"


def _get_openai_client() -> Any | None:
    """Create an OpenAI client if the package and API key are available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[gpt] OPENAI_API_KEY not set -- skipping chatgpt extraction.\n"
            "  Add to .env: OPENAI_API_KEY=sk-..."
        )
        return None

    if _openai_client_class is None:
        print("[gpt] openai not installed -- skipping.\n  pip install openai")
        return None

    return _openai_client_class(api_key=api_key)


def _strip_markdown_fences(content: str) -> str:
    """Remove common markdown fences around model JSON output."""
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    return re.sub(r"```$", "", content).strip()


def _extract_json_object(content: str) -> str:
    """Extract the first JSON object from a model response."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return match.group(0)
    return content


def _parse_phenotypes(content: str) -> list[str]:
    """Parse phenotype strings from model JSON output."""
    cleaned = _strip_markdown_fences(content)
    json_text = _extract_json_object(cleaned)
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        return []

    phenotypes = parsed.get("phenotypes", [])
    if not isinstance(phenotypes, list):
        return []

    return [item for item in phenotypes[:20] if isinstance(item, str)]


def _request_phenotypes(
    raw_text: str,
    model: str,
    skip_negated: bool,
) -> list[str]:
    """Request phenotype phrases from OpenAI and parse the JSON response."""
    client = _get_openai_client()
    if client is None:
        return []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHATGPT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(raw_text, skip_negated),
                },
            ],
            temperature=0,
            max_tokens=800,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[gpt] OpenAI request failed: {exc}")
        return []

    try:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return []
        return _parse_phenotypes(content.strip())
    except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[gpt] Could not parse model response: {exc}")
        return []


def _find_hpo_id(
    phrase_norm: str,
    lookup: dict[str, str],
) -> str | None:
    """Find an HPO ID for a normalized phenotype phrase."""
    hpo_id = lookup.get(phrase_norm)
    if hpo_id:
        return hpo_id

    if len(phrase_norm) < 8:
        return None

    for label_norm, candidate_id in lookup.items():
        if _is_conservative_match(phrase_norm, label_norm):
            return candidate_id

    return None


def _is_conservative_match(phrase_norm: str, label_norm: str) -> bool:
    """Return whether a phrase and label are close enough for fallback matching."""
    if len(label_norm) < 8:
        return False

    return (
        phrase_norm == label_norm
        or phrase_norm in label_norm
        or label_norm in phrase_norm
    )


def _build_extraction_result(
    hpo_id: str,
    phrase: str,
    hpo_labels: dict[str, str],
) -> ExtractionResult:
    """Build one ChatGPT extraction result."""
    return ExtractionResult(
        hpo_id=hpo_id,
        label=hpo_labels.get(hpo_id) or hpo_id,
        matched_text=phrase.strip(),
        method=ExtractionMethod.CHATGPT,
        confidence=0.85,
        start=None,
        end=None,
        negated=False,
    )


def extract_chatgpt(
    raw_text: str,
    hpo_labels: dict[str, str],
    skip_negated: bool = True,
    model: str = _CHATGPT_MODEL,
) -> list[ExtractionResult]:
    """
    HPO extraction using GPT-4o-mini.

    Args:
        raw_text: Raw clinical patient text.
        hpo_labels: Dict mapping HPO ID to label string.
        skip_negated: If True, ask the model to skip negated findings.
        model: OpenAI model identifier.

    Returns:
        Extraction results for each mapped HPO term.
    """
    phenotypes = _request_phenotypes(raw_text, model, skip_negated)
    lookup = build_label_lookup(hpo_labels)
    results: list[ExtractionResult] = []
    seen: set[str] = set()

    for phrase in phenotypes:
        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            continue

        hpo_id = _find_hpo_id(phrase_norm, lookup)
        if hpo_id is None:
            continue

        if hpo_id in HPO_BLOCKLIST or hpo_id in seen:
            continue

        seen.add(hpo_id)
        results.append(_build_extraction_result(hpo_id, phrase, hpo_labels))

    return results
