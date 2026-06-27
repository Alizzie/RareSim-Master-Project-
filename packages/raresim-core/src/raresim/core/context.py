"""
Application context for loading shared data like disease profiles, HPO labels, IC values,
and ancestors. This centralizes data access and ensures consistency across different
similarity methods and pipelines.
"""

from dataclasses import dataclass
from raresim.utils.io import load_json
from raresim.utils.paths import (
    HPO_LABELS_PATH,
    HPO_ANCESTORS_PATH,
    HPO_PARENTS_PATH,
    INFORMATION_CONTENT_PATH,
    DISEASE_METADATA_INDEX_PATH,
    DISEASE_ANCESTORS_PATH,
    DISEASE_PROFILES_PATH,
    CAN_DISEASE_PROFILES_PATH,
    ALIAS_TO_CANONICAL_PATH,
)
from raresim.types.schemas import PatientProfile
from raresim.types.result import AppMetadata


@dataclass
class AppContext:
    """Holds shared data and metadata for the application."""

    disease_profiles: dict[str, dict]
    hpo_labels: dict[str, str]
    ic_values: dict[str, float]
    ancestors: dict[str, list[str]]
    disease_ancestors: dict[str, list[str]]
    disease_metadata_index: dict[str, dict]
    hpo_parents: dict[str, list[str]]
    alias_to_canonical: dict[str, str]
    app_metadata: AppMetadata

    @classmethod
    def load(
        cls, patient: PatientProfile, use_canonical_profiles: bool = True
    ) -> "AppContext":
        """Load shared data and compute metadata based on the patient profile."""

        profile_file = (
            CAN_DISEASE_PROFILES_PATH
            if use_canonical_profiles
            else DISEASE_PROFILES_PATH
        )

        print("Loading shared data...")
        disease_profiles = load_json(profile_file)
        hpo_labels = load_json(HPO_LABELS_PATH)
        ic_values = load_json(INFORMATION_CONTENT_PATH)
        ancestors = load_json(HPO_ANCESTORS_PATH)
        disease_ancestors = load_json(DISEASE_ANCESTORS_PATH)
        disease_metadata_index = load_json(DISEASE_METADATA_INDEX_PATH)
        hpo_parents = load_json(HPO_PARENTS_PATH)
        alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)

        unfound_terms = [
            term for term in patient.hpo_terms if term not in hpo_labels.keys()
        ]

        if unfound_terms:
            print(
                f"  Warning: {len(unfound_terms)} patient term(s) not found in HPO labels."
            )

        app_metadata = AppMetadata(
            n_hpo_labels=len(hpo_labels.keys()),
            n_disease_profiles=len(disease_profiles.keys()),
            unfound_patient_terms=unfound_terms,
        )

        return cls(
            disease_profiles=disease_profiles,
            hpo_labels=hpo_labels,
            ic_values=ic_values,
            ancestors=ancestors,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
            hpo_parents=hpo_parents,
            alias_to_canonical=alias_to_canonical,
            app_metadata=app_metadata,
        )
