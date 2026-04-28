import math
from typing import Dict

from core.schemas import DiseaseProfile

"""Functions to compute information content (IC) values for HPO terms based on their frequencies across disease profiles."""


def compute_term_frequencies(
    disease_profiles: Dict[str, DiseaseProfile],
    use_propagated_terms: bool = True,
) -> Dict[str, int]:
    """
    Count in how many disease profiles each HPO term appears.
    """
    frequencies: Dict[str, int] = {}

    for profile in disease_profiles.values():
        terms = (
            profile.propagated_hpo_terms if use_propagated_terms else profile.hpo_terms
        )
        for term in terms:
            frequencies[term] = frequencies.get(term, 0) + 1

    return frequencies


def compute_information_content(
    term_frequencies: Dict[str, int],
    total_diseases: int,
) -> Dict[str, float]:
    """
    IC(term) = -log(freq(term) / total_diseases)
    """
    ic_values: Dict[str, float] = {}

    for term, freq in term_frequencies.items():
        probability = freq / total_diseases
        ic_values[term] = -math.log(probability)

    return ic_values
