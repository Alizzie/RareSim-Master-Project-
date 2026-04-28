def extend_explaination(
    explaination: dict, patient_terms: set, disease_terms: set
) -> dict:
    """Add shared term information to the explanation dictionary."""

    explaination["top_shared_terms"] = list(patient_terms.intersection(disease_terms))[
        :10
    ]
    return explaination
