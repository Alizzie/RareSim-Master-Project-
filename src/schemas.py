from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class DiseaseProfile:
    disease_id: str
    label: str
    source_ids: Dict[str, str] = field(default_factory=dict)
    hpo_terms: Set[str] = field(default_factory=set)
    propagated_hpo_terms: Set[str] = field(default_factory=set)

    ordo_label: Optional[str] = None
    ordo_description: Optional[str] = None

    mondo_label: Optional[str] = None
    mondo_description: Optional[str] = None

    hoom_label: Optional[str] = None
    hoom_description: Optional[str] = None

    merged_description: Optional[str] = None


@dataclass
class PatientProfile:
    patient_id: str
    raw_text: Optional[str] = None
    hpo_terms: Set[str] = field(default_factory=set)
    propagated_hpo_terms: Set[str] = field(default_factory=set)
    