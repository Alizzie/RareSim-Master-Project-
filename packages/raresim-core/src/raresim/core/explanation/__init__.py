"""
core.explanation — shared explanation schema and base builders.

Public API
----------
Schema types:
    ExplanationBlock
    HpoCoverageBlock, TokenCoverageBlock, CoverageBlock (union alias)
    TermMatch, TermEntry          — HPO term types
    TokenMatch, TokenEntry        — text token types
    MatchedTerm, UnmatchedTerm    — union aliases

Base builders:
    build_base_explanation        — ExplanationBlock for HPO methods
    build_coverage_block          — HpoCoverageBlock from HPO term sets
    build_token_coverage_block    — TokenCoverageBlock from token vectors
    build_matched_terms           — list[TermMatch] for HPO methods
    build_unmatched_terms         — list[TermEntry] for HPO methods
    build_matched_tokens          — list[TokenMatch] for text/hybrid modes
    build_unmatched_tokens        — list[TokenEntry] for text/hybrid modes
    build_ic_filter_block         — IC filtering diagnostics dict

Summary generators:
    set_based_summary, semantic_summary, tfidf_summary, embedding_summary
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
from raresim.core.explanation.summary import (
    set_based_summary,
    semantic_summary,
    tfidf_summary,
    embedding_summary,
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
    # shared
    "build_ic_filter_block",
    # summaries
    "set_based_summary",
    "semantic_summary",
    "tfidf_summary",
    "embedding_summary",
]
