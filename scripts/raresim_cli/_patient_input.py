"""
Patient input resolution for the RareSim CLI.

Handles all five input modes (text / hpo / patient file / defaults / interactive)
and returns a PatientProfile. Extracting this from app.py keeps the main flow
readable — app.py just calls resolve_patient() and gets a patient back.
"""

from raresim.hpo_extraction import build_patient_profile
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import get_ancestors_inclusive, preprocess_ancestor_sets
from raresim.utils.io import load_json, save_json
from raresim.utils.patient_loader import load_patient_with_extraction
from raresim.utils.paths import (
    HPO_ANCESTORS_PATH,
)
from raresim.similarity_methods.registry import DEFAULTS
from _utils import RARESIM_CLI_DIR


def _from_text(
    text: str, hpo_labels: dict, extraction_methods: list[str]
) -> PatientProfile:
    """Mode 1: raw clinical text → extract HPO terms → PatientProfile."""
    print("\nInput mode : raw text")
    print(f"Extraction : {extraction_methods}")
    print("\nExtracting HPO terms...")

    patient_dict, extracted_terms = build_patient_profile(
        patient_id="text_input_patient",
        raw_text=text,
        hpo_labels=hpo_labels,
        methods=extraction_methods,
    )

    print(f"\nExtracted {len(patient_dict['hpo_terms'])} HPO terms:")
    for t in extracted_terms:
        print(f"  {t['hpo_id']} | {t['label']} | method={t['method']}")

    if not patient_dict["hpo_terms"]:
        raise ValueError(
            "No HPO terms extracted — check your text or try different extraction methods."
        )

    tmp_path = RARESIM_CLI_DIR / "extracted_patient.json"
    save_json(patient_dict, tmp_path)
    return load_patient_with_extraction(tmp_path, hpo_labels)


def _from_hpo(hpo_arg: str, hpo_labels: dict) -> PatientProfile:
    """Mode 2: comma-separated HPO IDs → PatientProfile with propagation."""
    hpo_terms = [t.strip() for t in hpo_arg.split(",") if t.strip()]
    print("\nInput mode : HPO terms")
    print(f"HPO terms  : {hpo_terms}")

    ancestors = load_json(HPO_ANCESTORS_PATH)
    ancestor_sets = preprocess_ancestor_sets(ancestors)

    propagated: set[str] = set()
    for term in hpo_terms:
        propagated |= get_ancestors_inclusive(term, ancestor_sets)

    patient_dict = {
        "patient_id": "hpo_input_patient",
        "raw_text": "",
        "hpo_terms": hpo_terms,
        "propagated_hpo_terms": sorted(propagated),
        "methods_used": ["direct_input"],
    }
    tmp_path = RARESIM_CLI_DIR / "hpo_input_patient.json"
    save_json(patient_dict, tmp_path)
    return load_patient_with_extraction(tmp_path, hpo_labels)


def resolve_patient(args, hpo_labels: dict, prompt_fn) -> PatientProfile:
    """
    Resolve a PatientProfile from CLI args, dispatching on input mode.

    Modes (mutually exclusive in the arg parser):
        --text     → extract from clinical text
        --hpo      → direct HPO term IDs
        --patient  → load from JSON file
        --defaults → example patient
        (none)     → interactive prompt via prompt_fn
    """
    if args.text:
        return _from_text(args.text, hpo_labels, args.extraction_methods)

    if args.hpo:
        return _from_hpo(args.hpo, hpo_labels)

    if args.patient:
        print("\nInput mode : patient file")
        print(f"Patient    : {args.patient.name}")
        return load_patient_with_extraction(args.patient, hpo_labels)

    if args.defaults:
        print("\nInput mode : default example patient")
        return load_patient_with_extraction(DEFAULTS["patient_path"], hpo_labels)

    # No input flag → interactive prompt
    return prompt_fn(DEFAULTS, hpo_labels)
