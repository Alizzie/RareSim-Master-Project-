"""
Load ontology and annotation files from the internet to the local directory.
All files are saved in the "ontologies/model" directory.
If a file already exists, it will skip the download.
"""

from pathlib import Path
import requests


def download_file(url, save_path):
    """
    Download a file from a URL and save it to a local path.
    If the file already exists, it will skip the download.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"{save_path} already exists, skipping download.")
        return

    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"Saved to {save_path}")


def main():
    """
    Download all ontology and annotation files from the internet
    to the local directory.
    """
    files = {
        "hpo": {
            "url": "https://purl.obolibrary.org/obo/hp.owl",
            "path": "ontologies/model/hpo.owl",
        },
        "mondo_rare": {
            "url": "https://purl.obolibrary.org/obo/mondo/subsets/mondo-rare.owl",
            "path": "ontologies/model/mondo_rare.owl",
        },
        "ordo": {
            "url": "https://www.orphadata.com/data/ontologies/ordo/last_version/ORDO_en_4.8.owl",
            "path": "ontologies/model/ordo.owl",
        },
        "hoom": {
            "url": "https://data.bioontology.org/ontologies/HOOM/submissions/13/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
            "path": "ontologies/model/hoom.owl",
        },
        "phenotype_hpoa": {
            "url": "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-10-09/phenotype.hpoa",
            "path": "ontologies/model/phenotype.hpoa",
        },
        "orphadata_product4": {
            "url": "https://www.orphadata.com/data/xml/en_product4.xml",
            "path": "ontologies/model/en_product4_HPO.xml",
        },
        "monarch_disease_hpo": {
            "url": "https://data.monarchinitiative.org/monarch-kg/latest/tsv/all_associations/disease_to_phenotypic_feature_association.all.tsv.gz",
            "path": "ontologies/model/disease_to_phenotypic_feature_association.all.tsv.gz",
        },
    }

    for name, file_info in files.items():
        print(f"\n=== {name} ===")
        download_file(file_info["url"], file_info["path"])


if __name__ == "__main__":
    main()
