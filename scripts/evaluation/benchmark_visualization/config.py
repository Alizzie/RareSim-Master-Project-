"""
Configuration for RareSim evaluation visualization scripts.

Edit this file when you add new methods or validation tools.
"""


METHOD_LABELS = {
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "SapBERT",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
    "dmis-lab/biobert-v1.1": "BioBERT",
    "emilyalsentzer/Bio_ClinicalBERT": "ClinicalBERT",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "semantic_resnik_bma": "Resnik BMA",
    "semantic_lin_bma": "Lin BMA",
    "semantic_jiang_conrath_bma": "Jiang-Conrath BMA",
    "set_cosine": "Set cosine",
    "set_dice": "Set Dice",
    "set_jaccard": "Set Jaccard",
    "set_overlap": "Set overlap",
    "tfidf": "TF-IDF cosine",
    "tfidf_cosine": "TF-IDF cosine",
    "hpo2vec": "HPO2Vec",
    "denoising_autoencoder": "Denoising AE",
    "ensemble_rrf": "RRF ensemble",
    "ensemble_rrf_top": "RRF top ensemble",
    "ensemble_rrf_weighted": "Weighted RRF",
}


VALIDATION_TOOL_LABELS = {
    "dx29": "Dx29",
    "dx29_phrank": "Dx29 + Phrank",
    "lirical": "LIRICAL",
    "phenobrain": "PhenoBrain",
    "phenomizer": "Phenomizer",
}


DATASET_NAME_MAP = {
    "hms": "HMS",
    "lirical": "LIRICAL",
    "mme": "MME",
    "pumch_l": "PUMCH_L",
    "pumch-adm": "PUMCH-ADM",
    "ramedis": "RAMEDIS",
}


# Recall cut-offs used to draw the ranking curve (Q2).
RECALL_COLUMNS = ["R@1", "R@3", "R@5", "R@10"]

# ---------------------------------------------------------------------------
# Presentation settings
# ---------------------------------------------------------------------------

# Only these datasets are kept and shown, in this display order.
DATASETS = ["HMS", "MME", "LIRICAL", "RAMEDIS", "PUMCH_L", "PUMCH-ADM"]

# Columns shown (in this order) in the per-dataset metric tables and report.
SUMMARY_METRIC_COLUMNS = ["R@1", "R@3", "R@5", "R@10", "MRR", "NDCG@10"]

# Stable colours per system type
SYSTEM_TYPE_COLORS = {
    "RareSim method": "#2a7f9e",   # blue-teal
    "Ensemble": "#6a4c93",         # purple
    "Validation tool": "#d1603d",  # terracotta
}

# One distinct colour per dataset for grouped/series charts.
DATASET_COLORS = {
    "HMS": "#264653",
    "MME": "#2a9d8f",
    "LIRICAL": "#e9c46a",
    "RAMEDIS": "#f4a261",
    "PUMCH_L": "#e76f51",
    "PUMCH-ADM": "#8ab17d",
}
