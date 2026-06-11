"""
GPT extractor — HPT extraction via GPT-4o-mini.

The model extracts phenotype phrases from clinical text. Phrases are then
mapped locally to HPO IDs via hpo_labels — the model never generates
ontology identifiers directly, reducing hallucination risk.

Requires:
    pip install openai
    OPENAI_API_KEY=sk-... in .env file

Paper: https://ieeexplore.ieee.org/document/10340611
"""

import json
import os
import re
from typing import Dict, List

from raresim.hpo_extraction._config import HPO_BLOCKLIST

from ._types import ExtractionMethod, ExtractionResult
from ._utils import build_label_lookup, normalize_text

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


def extract_chatgpt(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
    model: str = _CHATGPT_MODEL,
) -> List[ExtractionResult]:
    """
    HPO extraction using GPT-4o-mini.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        skip_negated:  Unused — GPT prompt instructs model to skip negated findings.
        model:         OpenAI model identifier.

    Returns:
        List of ExtractionResult for each mapped HPO term.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[gpt] OPENAI_API_KEY not set -- skipping chatgpt extraction.\n"
            "  Add to .env: OPENAI_API_KEY=sk-..."
        )
        return []

    try:
        from openai import OpenAI
    except ImportError:
        print("[gpt] openai not installed -- skipping.\n  pip install openai")
        return []

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHATGPT_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"```$", "", content).strip()

        # Extract the JSON object if the model adds surrounding text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        parsed = json.loads(content)
        phenotypes = parsed.get("phenotypes", [])

    except Exception as e:
        print(f"[gpt] Extraction failed: {e}")
        return []

    if not isinstance(phenotypes, list):
        return []

    lookup = build_label_lookup(hpo_labels)
    results = []
    seen: set = set()

    for phrase in phenotypes[:20]:
        if not isinstance(phrase, str):
            continue

        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            continue

        hpo_id = lookup.get(phrase_norm)

        # Conservative fallback: containment match for reasonably specific phrases
        if not hpo_id and len(phrase_norm) >= 8:
            for label_norm, candidate_id in lookup.items():
                if len(label_norm) < 8:
                    continue
                if (
                    phrase_norm == label_norm
                    or phrase_norm in label_norm
                    or label_norm in phrase_norm
                ):
                    hpo_id = candidate_id
                    break

        if not hpo_id:
            continue
        if hpo_id in HPO_BLOCKLIST:
            continue
        if hpo_id in seen:
            continue

        seen.add(hpo_id)
        results.append(
            ExtractionResult(
                hpo_id=hpo_id,
                label=hpo_labels.get(hpo_id, hpo_id),
                matched_text=phrase.strip(),
                method=ExtractionMethod.CHATGPT,
                confidence=0.85,
                start=None,
                end=None,
                negated=False,
            )
        )

    return results
