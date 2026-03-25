import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


OWL_NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "obo": "http://purl.obolibrary.org/obo/",
    "IAO": "http://purl.obolibrary.org/obo/IAO_",
}


def _get_about_id(class_elem: ET.Element) -> Optional[str]:
    return class_elem.attrib.get(f"{{{OWL_NS['rdf']}}}about")


def _extract_label(class_elem: ET.Element) -> Optional[str]:
    label_elem = class_elem.find("rdfs:label", OWL_NS)
    if label_elem is not None and label_elem.text:
        return label_elem.text.strip()
    return None


def _extract_description(class_elem: ET.Element) -> Optional[str]:
    # Common places where ontology descriptions/definitions appear
    candidates = [
        class_elem.find("obo:IAO_0000115", OWL_NS),
        class_elem.find("IAO:0000115", OWL_NS),
        class_elem.find("rdfs:comment", OWL_NS),
    ]
    for elem in candidates:
        if elem is not None and elem.text:
            return elem.text.strip()
    return None


def _extract_xrefs(class_elem: ET.Element) -> List[str]:
    xrefs = []
    for elem in class_elem.findall("oboInOwl:hasDbXref", OWL_NS):
        if elem.text:
            xrefs.append(elem.text.strip())
    return xrefs


def load_hpo_owl(hpo_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    tree = ET.parse(hpo_path)
    root = tree.getroot()

    hpo_labels: Dict[str, str] = {}
    hpo_parents: Dict[str, Set[str]] = {}

    for cls in root.findall(".//owl:Class", OWL_NS):
        about = _get_about_id(cls)
        if not about or "HP_" not in about:
            continue

        hpo_id = about.split("/")[-1].replace("_", ":")
        label = _extract_label(cls)

        if label:
            hpo_labels[hpo_id] = label

        parents: Set[str] = set()
        for parent_elem in cls.findall("rdfs:subClassOf", OWL_NS):
            parent_ref = parent_elem.attrib.get(f"{{{OWL_NS['rdf']}}}resource", "")
            if "HP_" in parent_ref:
                parent_id = parent_ref.split("/")[-1].replace("_", ":")
                parents.add(parent_id)

        hpo_parents[hpo_id] = parents

    return hpo_labels, hpo_parents


def load_hpoa_annotations(hpoa_path: Path) -> List[dict]:
    records: List[dict] = []

    with hpoa_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]

    reader = csv.reader(lines, delimiter="\t")
    for row in reader:
        if len(row) < 4:
            continue

        records.append(
            {
                "database_id": row[0].strip(),
                "disease_name": row[1].strip(),
                "qualifier": row[2].strip(),
                "hpo_id": row[3].strip(),
            }
        )

    return records


def load_disease_ontology_metadata(
    ontology_path: Path,
    id_prefixes: tuple[str, ...],
) -> Dict[str, dict]:
    """
    Generic OWL metadata loader for disease ontologies.
    Extracts label, description, and xrefs for classes whose URI contains
    one of the provided prefixes, e.g. ('Orphanet_', 'MONDO_').
    """
    tree = ET.parse(ontology_path)
    root = tree.getroot()

    results: Dict[str, dict] = {}

    for cls in root.findall(".//owl:Class", OWL_NS):
        about = _get_about_id(cls)
        if not about:
            continue

        if not any(prefix in about for prefix in id_prefixes):
            continue

        local_id = about.split("/")[-1]

        label = _extract_label(cls)
        description = _extract_description(cls)
        xrefs = _extract_xrefs(cls)

        results[local_id] = {
            "uri": about,
            "label": label,
            "description": description,
            "xrefs": xrefs,
        }

    return results


def load_ordo_metadata(ordo_path: Path) -> Dict[str, dict]:
    return load_disease_ontology_metadata(
        ontology_path=ordo_path,
        id_prefixes=("Orphanet_", "ORDO_"),
    )


def load_mondo_metadata(mondo_path: Path) -> Dict[str, dict]:
    return load_disease_ontology_metadata(
        ontology_path=mondo_path,
        id_prefixes=("MONDO_",),
    )


def load_hoom_metadata(hoom_path: Path) -> Dict[str, dict]:
    return load_disease_ontology_metadata(
        ontology_path=hoom_path,
        id_prefixes=("HOOM_", "Orphanet_", "MONDO_"),
    )
