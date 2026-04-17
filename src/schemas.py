from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class DiseaseProfile:
    disease_id: str
    label: str

    hpo_terms: Set[str] = field(default_factory=set)
    propagated_hpo_terms: Set[str] = field(default_factory=set)

    ordo_label: Optional[str] = None
    ordo_description: Optional[str] = None
    mondo_label: Optional[str] = None
    mondo_description: Optional[str] = None
    hoom_label: Optional[str] = None
    hoom_description: Optional[str] = None
    merged_description: Optional[str] = None

    source_ids: Dict[str, object] = field(default_factory=dict)
    canonicalized_to_orpha: bool = False

    term_provenance: Dict[str, dict] = field(default_factory=dict)
    negative_hpo_terms: Set[str] = field(default_factory=set)


@dataclass
class PatientProfile:
    patient_id: str
    raw_text: str
    hpo_terms: Set[str] = field(default_factory=set)
    propagated_hpo_terms: Set[str] = field(default_factory=set)
