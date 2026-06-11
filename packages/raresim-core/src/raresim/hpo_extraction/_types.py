"""
Data types for HPO extraction.

ExtractionMethod : enum of all supported extraction backends.
ExtractionResult : single extracted HPO term with full provenance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExtractionMethod(str, Enum):
    DICTIONARY     = "hpo_label_dictionary_match"
    BIOMEDICAL_NER = "biomedical_ner_d4data"
    FAST_HPO_CR    = "fast_hpo_cr"
    CHATGPT        = "chatgpt_extraction"
    PHENOBRAIN_API = "phenobrain_api"


@dataclass
class ExtractionResult:
    hpo_id: str
    label: str
    matched_text: str
    method: ExtractionMethod
    confidence: float = 1.0
    start: Optional[int] = None
    end: Optional[int] = None
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "hpo_id":       self.hpo_id,
            "label":        self.label,
            "matched_text": self.matched_text,
            "method":       self.method.value,
            "confidence":   self.confidence,
            "start":        self.start,
            "end":          self.end,
            "negated":      self.negated,
        }
        