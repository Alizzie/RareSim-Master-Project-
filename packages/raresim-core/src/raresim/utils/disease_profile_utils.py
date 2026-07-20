"""Utility functions for disease profiles."""


def disease_exclusion_inputs(profile: dict) -> tuple[set[str], set[str]]:
    """Raw terms and negative terms for exclusion-conflict detection."""
    return (
        set(profile.get("hpo_terms", [])),
        set(profile.get("negative_hpo_terms", [])),
    )
