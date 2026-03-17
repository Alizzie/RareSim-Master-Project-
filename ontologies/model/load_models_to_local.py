"""
Load all ontology files from the internet to the local directory.
All files should be saved in the "ontologies/model" directory with the file extension ".owl".
If the file already exists, it will skip the download.
"""

from pathlib import Path
import requests


def download_ontology(url, save_path):
    """
    Download an ontology from a URL and save it to a local path.
    If the file already exists, it will skip the download.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"{save_path} already exists, skipping download.")
        return

    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print(f"Saved to {save_path}")


def main():
    """
    Download all ontology files from the internet to the local directory.
    """

    urls = {
        "hpo": "https://purl.obolibrary.org/obo/hp.owl",
        "mondo_rare": "https://purl.obolibrary.org/obo/mondo/subsets/mondo-rare.owl",
        "ordo": "https://www.orphadata.com/data/ontologies/ordo/last_version/ORDO_en_4.8.owl",
        "hoom": "https://data.bioontology.org/ontologies/HOOM/submissions/13/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
        "phenotype.hpoa": "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-10-09/phenotype.hpoa"
    }

    for name, url in urls.items():
        download_ontology(url, f"ontologies/model/{name}.owl")


if __name__ == "__main__":
    main()
