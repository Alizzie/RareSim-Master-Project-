"""
Application context for loading shared data like disease profiles, HPO labels, IC values,
and ancestors. This centralizes data access and ensures consistency across different
similarity methods and pipelines.
"""

from dataclasses import dataclass
from raresim.utils.io import load_json
from raresim.utils.paths import ARTIFACTS_DIR
from raresim.types.schemas import PatientProfile
from raresim.types.result import AppMetadata


@dataclass
class AppContext:
    """Holds shared data and metadata for the application."""

    disease_profiles: dict[str, dict]
    hpo_labels: dict[str, str]
    ic_values: dict[str, float]
    ancestors: dict[str, list[str]]
    app_metadata: AppMetadata

    @classmethod
    def load(
        cls, patient: PatientProfile, use_canonical_profiles: bool = True
    ) -> "AppContext":
        """Load shared data and compute metadata based on the patient profile."""

        profile_file = (
            "canonical_disease_profiles.json"
            if use_canonical_profiles
            else "disease_profiles.json"
        )

        print("Loading shared data...")
        disease_profiles = load_json(ARTIFACTS_DIR / profile_file)
        hpo_labels = load_json(ARTIFACTS_DIR / "hpo_labels.json")
        ic_values = load_json(ARTIFACTS_DIR / "information_content.json")
        ancestors = load_json(ARTIFACTS_DIR / "hpo_ancestors.json")

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
            app_metadata=app_metadata,
        )
