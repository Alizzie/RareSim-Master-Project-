import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISEASE_PROFILES_PATH = PROJECT_ROOT / "outputs" / "shared" / "disease_profiles.json"


def load_disease_profiles(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_namespace(disease_id: str) -> str:
    if ":" in disease_id:
        return disease_id.split(":", maxsplit=1)[0]
    return "UNKNOWN"


def format_percent(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def summarize_profiles(disease_profiles: dict) -> None:
    total = len(disease_profiles)

    with_hpo = 0
    without_hpo = 0
    with_propagated = 0
    with_description = 0
    canonicalized_to_orpha = 0

    namespace_counter = Counter()
    namespace_with_hpo_counter = Counter()

    omim_with_hpo = 0
    orpha_with_hpo = 0
    mondo_with_hpo = 0
    decipher_with_hpo = 0

    for disease_id, profile in disease_profiles.items():
        hpo_terms = profile.get("hpo_terms", [])
        propagated_terms = profile.get("propagated_hpo_terms", [])
        description = profile.get("merged_description")
        namespace = get_namespace(disease_id)

        namespace_counter[namespace] += 1

        has_hpo = len(hpo_terms) > 0
        has_propagated = len(propagated_terms) > 0
        has_description = bool(description and description.strip())
        is_canonicalized = profile.get("canonicalized_to_orpha", False)

        if has_hpo:
            with_hpo += 1
            namespace_with_hpo_counter[namespace] += 1
        else:
            without_hpo += 1

        if has_propagated:
            with_propagated += 1

        if has_description:
            with_description += 1

        if is_canonicalized:
            canonicalized_to_orpha += 1

        if has_hpo and namespace == "OMIM":
            omim_with_hpo += 1
        elif has_hpo and namespace == "ORPHA":
            orpha_with_hpo += 1
        elif has_hpo and namespace == "MONDO":
            mondo_with_hpo += 1
        elif has_hpo and namespace == "DECIPHER":
            decipher_with_hpo += 1

    print("===== DISEASE PROFILE STATS =====")
    print(f"Total profiles: {total}")
    print(f"With HPO terms: {with_hpo} ({format_percent(with_hpo, total)})")
    print(
        f"Without HPO terms: {without_hpo} "
        f"({format_percent(without_hpo, total)})"
    )
    print(
        f"With propagated terms: {with_propagated} "
        f"({format_percent(with_propagated, total)})"
    )
    print(
        f"With description: {with_description} "
        f"({format_percent(with_description, total)})"
    )

    print("\n===== ORPHA CANONICALIZATION CHECK =====")
    print(
        f"Profiles with canonicalized_to_orpha = True: "
        f"{canonicalized_to_orpha} "
        f"({format_percent(canonicalized_to_orpha, total)})"
    )

    print("\n===== FULL NAMESPACE DISTRIBUTION =====")
    for namespace, count in sorted(
        namespace_counter.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{namespace}: {count} ({format_percent(count, total)})")

    print("\n===== HPO-ANNOTATED PROFILES BY NAMESPACE =====")
    print(f"OMIM entries with HPO terms: {omim_with_hpo}")
    print(f"ORPHA entries with HPO terms: {orpha_with_hpo}")
    print(f"MONDO entries with HPO terms: {mondo_with_hpo}")
    print(f"DECIPHER entries with HPO terms: {decipher_with_hpo}")

    print("\n===== FULL HPO-ANNOTATED NAMESPACE DISTRIBUTION =====")
    for namespace, count in sorted(
        namespace_with_hpo_counter.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{namespace}: {count} ({format_percent(count, with_hpo)})")

    print("\n===== QUICK SANITY CHECKS =====")
    if with_hpo == 0:
        print("WARNING: No diseases with HPO terms were found.")
    else:
        print("OK: HPO-annotated disease profiles are present.")

    if with_propagated < with_hpo:
        print(
            "WARNING: Some diseases have HPO terms but no propagated terms. "
            "Check propagation logic."
        )
    else:
        print("OK: Propagated terms look consistent with HPO terms.")

    if orpha_with_hpo < omim_with_hpo:
        print(
            "NOTE: OMIM still dominates the HPO-annotated subset. "
            "Canonicalization is still incomplete."
        )
    else:
        print(
            "OK: ORPHA is now dominant or comparable in the HPO-annotated subset."
        )


def main() -> None:
    disease_profiles = load_disease_profiles(DISEASE_PROFILES_PATH)
    summarize_profiles(disease_profiles)


if __name__ == "__main__":
    main()
    