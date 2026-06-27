"""Schemas for disease and patient profiles."""


from dataclasses import dataclass, field


@dataclass
class DiseaseProfile:  # pylint: disable=too-many-instance-attributes
    """A disease profile, including its HPO terms and metadata."""
    disease_id: str
    label: str
    profile_type: str | None = None

    hpo_terms: set[str] = field(default_factory=set)
    propagated_hpo_terms: set[str] = field(default_factory=set)

    ordo_label: str | None = None
    ordo_description: str | None = None
    mondo_label: str | None = None
    mondo_description: str | None = None
    hoom_label: str | None = None
    hoom_description: str | None = None
    merged_description: str | None = None

    source_ids: dict[str, str] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)
    category_source_id: str | None = None
    canonicalized_to_orpha: bool = False

    term_provenance: dict[str, dict] = field(default_factory=dict)
    negative_hpo_terms: set[str] = field(default_factory=set)


@dataclass
class PatientProfile:
    """A patient profile, including their HPO terms and metadata."""
    patient_id: str
    raw_text: str
    hpo_terms: set[str] = field(default_factory=set)
    propagated_hpo_terms: set[str] = field(default_factory=set)

    def get_terms(self, use_propagated: bool = True) -> set[str]:
        """Return the set of HPO terms for this patient, either propagated or not."""
        return self.propagated_hpo_terms if use_propagated else self.hpo_terms
