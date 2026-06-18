"""
core.explanation — shared explanation schema and base builders.

Public API
----------
Schema types (import for type hints or serialization):
    ExplanationBlock, CoverageBlock, TermMatch, TermEntry

Base builders (import in each method's explanation.py):
    build_base_explanation   — assembles a complete ExplanationBlock
    build_coverage_block     — CoverageBlock only
    build_matched_terms      — enriched matched-term list
    build_unmatched_terms    — enriched unmatched-term list
    build_ic_filter_block    — IC filtering diagnostics dict

Summary generators (import in each method's explanation.py):
    set_based_summary
    semantic_summary
    tfidf_summary
    embedding_summary
"""

from raresim.core.explanation.schema import (
    ExplanationBlock,
    CoverageBlock,
    TermMatch,
    TermEntry,
)
from raresim.core.explanation.base_explainer import (
    build_base_explanation,
    build_coverage_block,
    build_matched_terms,
    build_unmatched_terms,
    build_ic_filter_block,
)
from raresim.core.explanation.summary import (
    set_based_summary,
    semantic_summary,
    tfidf_summary,
    embedding_summary,
)

__all__ = [
    # schema
    "ExplanationBlock",
    "CoverageBlock",
    "TermMatch",
    "TermEntry",
    # builders
    "build_base_explanation",
    "build_coverage_block",
    "build_matched_terms",
    "build_unmatched_terms",
    "build_ic_filter_block",
    # summaries
    "set_based_summary",
    "semantic_summary",
    "tfidf_summary",
    "embedding_summary",
]
