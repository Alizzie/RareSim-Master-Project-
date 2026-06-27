"""Functions to load and parse ontology files (OWL) and annotation files
(HPOA and other disease->phenotype sources), extracting relevant metadata and
relationships.
"""

import csv
import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OWL_NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "obo": "http://purl.obolibrary.org/obo/",
    "IAO": "http://purl.obolibrary.org/obo/IAO_",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}


def _normalize_profile_type(value: str | None) -> str | None:
    """Normalize ORDO disease/category type labels."""
    if not value:
        return None

    cleaned = value.strip().lower()

    # Ignore identifiers. These are not profile types.
    if re.fullmatch(r"(orpha|orphanet|ordo)[:_]\d+", cleaned):
        return None

    if re.fullmatch(r"\d+", cleaned):
        return None

    if "category" in cleaned:
        return "category"
    if "group" in cleaned:
        return "disease_group"
    if "clinical subtype" in cleaned:
        return "clinical_subtype"
    if "etiological subtype" in cleaned or "aetiological subtype" in cleaned:
        return "etiological_subtype"
    if "histopathological subtype" in cleaned:
        return "histopathological_subtype"
    if "subtype" in cleaned:
        return "subtype"
    if "disorder" in cleaned or "disease" in cleaned:
        return "specific_disease"

    # Important: do not return unknown free text as a type.
    # Otherwise ORPHA IDs or unrelated annotations can leak into profile_type.
    return None


def _extract_profile_type(class_elem: ET.Element) -> str | None:
    """
    Extract ORDO profile type/category when available.

    This is intentionally conservative. Some ORDO skos:notation values are
    identifiers such as ORPHA:123 or Orphanet_123, not profile types. Those
    values must not leak into profile_type.
    """
    for elem in class_elem.findall("skos:notation", OWL_NS):
        if not elem.text:
            continue

        normalized = _normalize_profile_type(elem.text)
        if normalized:
            return normalized

    return None


def _extract_exact_matches(class_elem: ET.Element) -> list[str]:
    """Extract skos:exactMatch resource URIs from an ontology class.

    These matches are used for canonical ID mapping, not for profile_type.
    """
    matches: list[str] = []
    seen: set[str] = set()

    for elem in class_elem.findall("skos:exactMatch", OWL_NS):
        resource = elem.attrib.get(f"{{{OWL_NS['rdf']}}}resource")
        if not resource:
            continue

        resource = resource.strip()
        if resource and resource not in seen:
            matches.append(resource)
            seen.add(resource)

    return matches


def _local_name(tag: str) -> str:
    """Strip XML namespace and return the local tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _debug_print_xml_sample(root: ET.Element, max_items: int = 40) -> None:
    """Print a small sample of XML tags/text for debugging parser issues."""
    print("DEBUG XML sample:")
    for i, elem in enumerate(root.iter()):
        tag = _local_name(elem.tag)
        text = (elem.text or "").strip()
        print(f"{i:03d} TAG={tag} TEXT={text[:80]}")
        if i >= max_items:
            break


def _get_about_id(class_elem: ET.Element) -> str | None:
    return class_elem.attrib.get(f"{{{OWL_NS['rdf']}}}about")


def _extract_label(class_elem: ET.Element) -> str | None:
    label_elem = class_elem.find("rdfs:label", OWL_NS)
    if label_elem is not None and label_elem.text:
        return label_elem.text.strip()
    return None


def _extract_description(class_elem: ET.Element) -> str | None:
    candidates = [
        class_elem.find("obo:IAO_0000115", OWL_NS),
        class_elem.find("IAO:0000115", OWL_NS),
        class_elem.find("rdfs:comment", OWL_NS),
    ]
    for elem in candidates:
        if elem is not None and elem.text:
            return elem.text.strip()
    return None


def _extract_xrefs(class_elem: ET.Element) -> list[str]:
    xrefs = []
    for elem in class_elem.findall("oboInOwl:hasDbXref", OWL_NS):
        if elem.text:
            xrefs.append(elem.text.strip())
    return xrefs


def load_hpo_owl(hpo_path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    tree = ET.parse(hpo_path)
    root = tree.getroot()

    hpo_labels: dict[str, str] = {}
    hpo_parents: dict[str, set[str]] = {}

    for cls in root.findall(".//owl:Class", OWL_NS):
        about = _get_about_id(cls)
        if not about or "HP_" not in about:
            continue

        hpo_id = about.split("/")[-1].replace("_", ":")
        label = _extract_label(cls)

        if label:
            hpo_labels[hpo_id] = label

        parents: set[str] = set()
        for parent_elem in cls.findall("rdfs:subClassOf", OWL_NS):
            parent_ref = parent_elem.attrib.get(f"{{{OWL_NS['rdf']}}}resource", "")
            if "HP_" in parent_ref:
                parent_id = parent_ref.split("/")[-1].replace("_", ":")
                parents.add(parent_id)

        hpo_parents[hpo_id] = parents

    return hpo_labels, hpo_parents


def load_hpoa_annotations(hpoa_path: Path) -> list[dict]:
    """
    HPOA columns:
    database_id, disease_name, qualifier, hpo_id, reference, evidence,
    onset, frequency, sex, modifier, aspect, biocuration
    """
    records: list[dict] = []

    with hpoa_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]

    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        database_id = (row.get("database_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        qualifier = (row.get("qualifier") or "").strip()
        hpo_id = (row.get("hpo_id") or "").strip()
        frequency = (row.get("frequency") or "").strip()

        if not database_id or not hpo_id:
            continue

        records.append(
            {
                "database_id": database_id,
                "disease_name": disease_name,
                "qualifier": qualifier,
                "hpo_id": hpo_id,
                "frequency_code": frequency if frequency else None,
                "source": "HPOA",
            }
        )

    return records


def load_disease_ontology_metadata(
    ontology_path: Path,
    id_prefixes: tuple[str, ...],
    normalize_local_id_func,
) -> dict[str, dict]:
    tree = ET.parse(ontology_path)
    root = tree.getroot()

    results: dict[str, dict] = {}

    for cls in root.findall(".//owl:Class", OWL_NS):
        about = _get_about_id(cls)
        if not about:
            continue

        if not any(prefix in about for prefix in id_prefixes):
            continue

        local_id = about.split("/")[-1]

        results[local_id] = {
            "uri": about,
            "normalized_id": normalize_local_id_func(local_id),
            "label": _extract_label(cls),
            "description": _extract_description(cls),
            "xrefs": _extract_xrefs(cls),
            "exact_matches": _extract_exact_matches(cls),
            "profile_type": _extract_profile_type(cls),
        }

    return results


def load_ordo_parents(ordo_path: Path) -> dict[str, set[str]]:
    """
    Parse the ORDO OWL file and extract IS-A (subClassOf) parent relations.

    Mirrors load_hpo_owl but for Orphanet_ nodes.  Only direct subClassOf
    relations with an rdf:resource attribute are kept — anonymous Restriction
    blocks (part-of, has-part, etc.) are intentionally skipped because they
    carry part-of semantics, not IS-A.

    Returns:
        dict mapping ORPHA:NNNN → set of ORPHA:NNNN parents.
        Leaf nodes with no IS-A parents are included with an empty set so
        compute_ancestors can reach them.
    """
    tree = ET.parse(ordo_path)
    root = tree.getroot()

    ordo_parents: dict[str, set[str]] = {}

    for cls in root.findall(".//owl:Class", OWL_NS):
        about = _get_about_id(cls)
        if not about or "Orphanet_" not in about:
            continue

        local = about.split("/")[-1]  # "Orphanet_10"
        num = local.split("_")[-1]
        if not num.isdigit():
            continue
        orpha_id = f"ORPHA:{num}"

        parents: set[str] = set()
        for sub in cls.findall("rdfs:subClassOf", OWL_NS):
            # Only direct resource references — skip anonymous Restriction blocks
            ref = sub.attrib.get(f"{{{OWL_NS['rdf']}}}resource", "")
            if "Orphanet_" not in ref:
                continue
            parent_num = ref.split("Orphanet_")[-1]
            if parent_num.isdigit():
                parents.add(f"ORPHA:{parent_num}")

        ordo_parents[orpha_id] = parents

    return ordo_parents


def load_ordo_metadata(ordo_path: Path, normalize_local_id_func) -> dict[str, dict]:
    return load_disease_ontology_metadata(
        ontology_path=ordo_path,
        id_prefixes=("Orphanet_", "ORDO_"),
        normalize_local_id_func=normalize_local_id_func,
    )


def load_mondo_metadata(mondo_path: Path, normalize_local_id_func) -> dict[str, dict]:
    return load_disease_ontology_metadata(
        ontology_path=mondo_path,
        id_prefixes=("MONDO_",),
        normalize_local_id_func=normalize_local_id_func,
    )


def load_hoom_metadata(hoom_path: Path, normalize_local_id_func) -> dict[str, dict]:
    results = load_disease_ontology_metadata(
        ontology_path=hoom_path,
        id_prefixes=("HOOM_", "Orphanet_", "MONDO_"),
        normalize_local_id_func=normalize_local_id_func,
    )

    if not results:
        print("WARNING: HOOM metadata loader found 0 entries.")
        print("DEBUG: printing first 30 owl:Class rdf:about values from HOOM...")

        tree = ET.parse(hoom_path)
        root = tree.getroot()

        count = 0
        for cls in root.findall(".//owl:Class", OWL_NS):
            about = cls.attrib.get(f"{{{OWL_NS['rdf']}}}about")
            if about:
                print(about)
                count += 1
            if count >= 30:
                break

    return results


def load_hoom_hpo_annotations(hoom_path: Path) -> list[dict]:
    """
    Parse HOOM for disease -> HPO annotations.

    Supports compact labels like:
        Orpha:2632_HP:0100864_Freq:VF

    and frequency categories in the OWL axioms like:
        #VeryFrequent, #Frequent, #Occasional, #VeryRare, #Excluded
    """
    compact_pattern = re.compile(
        r"Orpha:(\d+)_HP:(\d+)_Freq:([A-Za-z0-9%]+)",
        flags=re.IGNORECASE,
    )

    tree = ET.parse(hoom_path)
    root = tree.getroot()

    records: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()

    for eq in root.findall(".//owl:EquivalentClasses", OWL_NS):
        class_elem = eq.find("owl:Class", OWL_NS)
        if class_elem is None:
            continue

        iri = class_elem.attrib.get("IRI", "")
        match = compact_pattern.search(iri)
        if not match:
            continue

        orpha_num, hp_num, compact_freq = match.groups()
        disease_id = f"ORPHA:{orpha_num}"
        hpo_id = f"HP:{hp_num.zfill(7)}"

        freq_value = compact_freq.upper()

        # Try to refine frequency from the OWL structure if present
        intersection = eq.find("owl:ObjectIntersectionOf", OWL_NS)
        if intersection is not None:
            for obj_some in intersection.findall("owl:ObjectSomeValuesFrom", OWL_NS):
                obj_prop = obj_some.find("owl:ObjectProperty", OWL_NS)
                cls = obj_some.find("owl:Class", OWL_NS)
                if obj_prop is None or cls is None:
                    continue

                prop_iri = obj_prop.attrib.get("IRI", "")
                class_iri = cls.attrib.get("IRI", "")

                if prop_iri == "#has_frequency":
                    freq_value = class_iri.lstrip("#").strip() or freq_value
                    break

        key = (disease_id, hpo_id, freq_value)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            {
                "database_id": disease_id,
                "disease_name": disease_id,
                "qualifier": "",
                "hpo_id": hpo_id,
                "frequency_code": freq_value,
                "source": "HOOM",
            }
        )

    if not records:
        print("WARNING: HOOM HPO annotation loader found 0 annotations.")
        _debug_print_xml_sample(root, max_items=60)

    return records


def load_orphadata_product4_annotations(product4_path: Path) -> list[dict]:
    tree = ET.parse(product4_path)
    root = tree.getroot()

    records: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()

    disorder_count = 0
    assoc_count = 0

    for disorder in root.iter():
        if _local_name(disorder.tag).lower() != "disorder":
            continue

        disorder_count += 1
        orpha_id = None

        # find OrphaCode within this Disorder
        for elem in disorder.iter():
            tag = _local_name(elem.tag).lower()
            text = (elem.text or "").strip()

            if tag == "orphacode" and text.isdigit():
                orpha_id = f"ORPHA:{text}"
                break

        if not orpha_id:
            continue

        # find phenotype associations within this Disorder
        for assoc in disorder.iter():
            if _local_name(assoc.tag).lower() != "hpodisorderassociation":
                continue

            assoc_count += 1
            hpo_id = None
            frequency_text = None

            for elem in assoc.iter():
                tag = _local_name(elem.tag).lower()
                text = (elem.text or "").strip()

                if tag == "hpoid" and text.startswith("HP:"):
                    hpo_id = text

                elif tag == "name" and text:
                    lower = text.lower()
                    if any(
                        phrase in lower
                        for phrase in (
                            "very rare",
                            "occasional",
                            "frequent",
                            "very frequent",
                            "obligate",
                            "excluded",
                        )
                    ):
                        frequency_text = text

            if not hpo_id:
                continue

            key = (orpha_id, hpo_id, frequency_text)
            if key in seen:
                continue
            seen.add(key)

            records.append(
                {
                    "database_id": orpha_id,
                    "disease_name": orpha_id,
                    "qualifier": "",
                    "hpo_id": hpo_id,
                    "frequency_code": frequency_text,
                    "source": "ORPHADATA_PRODUCT4",
                }
            )

    if not records:
        print("WARNING: Orphadata Product 4 loader found 0 annotations.")
        print(f"DEBUG: Number of <Disorder> blocks seen: {disorder_count}")
        print(f"DEBUG: Number of <HPODisorderAssociation> blocks seen: {assoc_count}")
        _debug_print_xml_sample(root, max_items=60)

    return records


def load_monarch_disease_hpo_annotations(monarch_path: Path) -> list[dict]:
    """
    Parse Monarch disease_to_phenotypic_feature_association TSV.gz.
    """
    records: list[dict] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()

    with gzip.open(monarch_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        for row in reader:
            subject = (row.get("subject") or "").strip()
            subject_label = (row.get("subject_label") or subject).strip()
            obj = (row.get("object") or "").strip()
            qualifier = (row.get("qualifier") or "").strip()

            if not subject or not obj:
                continue
            if not obj.startswith("HP:"):
                continue

            if not any(
                subject.startswith(prefix)
                for prefix in (
                    "MONDO:",
                    "OMIM:",
                    "ORPHA:",
                    "DECIPHER:",
                    "DOID:",
                )
            ):
                continue

            key = (subject, obj, qualifier or None, "MONARCH")
            if key in seen:
                continue
            seen.add(key)

            records.append(
                {
                    "database_id": subject,
                    "disease_name": subject_label,
                    "qualifier": qualifier,
                    "hpo_id": obj,
                    "frequency_code": None,
                    "source": "MONARCH",
                }
            )

    return records
