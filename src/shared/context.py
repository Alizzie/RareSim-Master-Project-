"""
Application conetxt for loading shared data like disease profiles, HPO labels, IC values, and ancestors.
This centralizes data access and ensures consistency across different similarity methods and pipelines.
"""

from dataclasses import dataclass
from shared.io import load_json
from shared.paths import SHARED_DIR
from shared.result import AppMetadata
from core.schemas import PatientProfile


@dataclass
class AppContext:
    disease_profiles: dict
    hpo_labels: dict
    ic_values: dict
    ancestors: dict
    app_metadata: AppMetadata

    @classmethod
    def load(
        cls, patient: PatientProfile, use_canonical_profiles: bool = True
    ) -> "AppContext":

        profile_file = (
            "canonical_disease_profiles.json"
            if use_canonical_profiles
            else "disease_profiles.json"
        )

        disease_profiles = (load_json(SHARED_DIR / profile_file),)
        hpo_labels = (load_json(SHARED_DIR / "hpo_labels.json"),)
        ic_values = (load_json(SHARED_DIR / "information_content.json"),)
        ancestors = (load_json(SHARED_DIR / "hpo_ancestors.json"),)

        unfound_terms = [
            term for term in patient.hpo_terms if term not in hpo_labels[0]
        ]

        print("Loading shared data...")

        app_metadata = AppMetadata(
            n_hpo_labels=len(hpo_labels[0]),
            n_disease_profiles=len(disease_profiles[0]),
            n_patient_terms=len(patient.hpo_terms),
            n_patient_propagated_terms=len(patient.propagated_hpo_terms),
            unfound_patient_terms=unfound_terms,
        )

        return cls(
            disease_profiles=disease_profiles[0],
            hpo_labels=hpo_labels[0],
            ic_values=ic_values[0],
            ancestors=ancestors[0],
            app_metadata=app_metadata,
        )
