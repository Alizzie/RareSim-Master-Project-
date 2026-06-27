"""
core.explanation — shared explanation schema and base builders.

This package provides the structural contract that all similarity methods
must satisfy. It does NOT provide summary string generators — those are
method-specific and live in each method's own explanation.py.

Public API
----------
Schema types:
    ExplanationBlock
    HpoCoverageBlock, TokenCoverageBlock, CoverageBlock (union alias)
    TermMatch, TermEntry          — HPO term types
    TokenMatch, TokenEntry        — text token types
    MatchedTerm, UnmatchedTerm    — union aliases

Base builders (for HPO methods):
    build_base_explanation        — assemble ExplanationBlock from HPO term sets
    build_coverage_block          — HpoCoverageBlock from two HPO term sets
    build_matched_terms           — list[TermMatch], sorted by IC descending
    build_unmatched_terms         — list[TermEntry], sorted by IC descending

Base builders (for token/text methods):
    build_base_token_explanation  — assemble ExplanationBlock from token vectors
    build_token_coverage_block    — TokenCoverageBlock from two token vectors
    build_matched_tokens          — list[TokenMatch], sorted by IDF descending
    build_unmatched_tokens        — list[TokenEntry], sorted by IDF descending

Shared diagnostics builder:
    build_ic_filter_block         — IC filtering impact dict for method_specific
"""

from raresim.core.explanation.schema import (
    ExplanationBlock,
    HpoCoverageBlock,
    TokenCoverageBlock,
    CoverageBlock,
    TermMatch,
    TermEntry,
    TokenMatch,
    TokenEntry,
    MatchedTerm,
    UnmatchedTerm,
)
from raresim.core.explanation.base_explainer import (
    build_base_explanation,
    build_base_token_explanation,
    build_coverage_block,
    build_token_coverage_block,
    build_matched_terms,
    build_unmatched_terms,
    build_matched_tokens,
    build_unmatched_tokens,
    build_ic_filter_block,
)

__all__ = [
    # schema — HPO types
    "ExplanationBlock",
    "HpoCoverageBlock",
    "TokenCoverageBlock",
    "CoverageBlock",
    "TermMatch",
    "TermEntry",
    # schema — token types
    "TokenMatch",
    "TokenEntry",
    "MatchedTerm",
    "UnmatchedTerm",
    # builders — HPO
    "build_base_explanation",
    "build_coverage_block",
    "build_matched_terms",
    "build_unmatched_terms",
    # builders — token
    "build_base_token_explanation",
    "build_token_coverage_block",
    "build_matched_tokens",
    "build_unmatched_tokens",
    # shared diagnostics
    "build_ic_filter_block",
]
