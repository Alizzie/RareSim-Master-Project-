"""
Pipeline:
  1. Build a graph from HPO parents and disease profiles
  2. Run IC weighted random walks from every node
  3. Train Word2Vec on the walks
  4. For patient: aggregate their HPO term embeddings into 1 patient vector
  5. For each disease: same as 4 but for HPO
  6. Rank diseases by some similarity to the patient vector
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gensim.models import Word2Vec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
HPO2VEC_DIR = PROJECT_ROOT / "outputs" / "hpo2vec"
HPO2VEC_DIR.mkdir(parents=True, exist_ok=True)


# Parameters
# Controls the random walk and Word2Vec training behaviour
# My defaults

WALK_LENGTH = 80        # number of steps per random walk
WALKS_PER_NODE = 10     # how many walks to start from each node
P = 1.0                 # return param: controls likelihood of revisiting a node
                        # lower p = more likely to backtrack 
Q = 0.5                 # in out parameter: controls BFS vs DFS behaviour
                        # q < 1 = biased toward DFS
                        # q > 1 = toward BFS
EMBEDDING_DIM = 128     # dimension of the learned embeddings
WINDOW_SIZE = 10        # Word2Vec context window aka how many neighbours to consider
MIN_COUNT = 1           # minimum term frequency to include in vocabulary
WORKERS = 4             # parallel workers for Word2Vec training
EPOCHS = 5              # training epochs



# Load shared artifacts

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)



# Step 1L Build the graph

# Just an adjacency list built from two sources:
#   - hpo_parents.json: IS_A edges (child -> parent HPO term)
#   - disease_profiles.json: HAS_PHENOTYPE edges (disease -> HPO term)
#
# Both diseases and HPO terms are nodes, all in the same graph
# A walk can hop from a disease to its phenotypes and then up the HPO hierarchy

def build_graph(
    hpo_parents: Dict[str, List[str]],
    disease_profiles: Dict[str, dict],
    use_propagated: bool = False,
) -> Dict[str, List[str]]:
    """
    Builds {node_id: [neighbour_id, ...]} from HPO parents and disease profiles
    All edges are bidirectional so walks can go up/down the hierarchy
    Use raw hpo_terms (not propagated) since IS_A edges already capture
    the hierarchy, otherwise redundant
    """
    graph: Dict[str, List[str]] = {}

    def add_edge(a: str, b: str):
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    # IS_A 
    for child, parents in hpo_parents.items():
        for parent in parents:
            add_edge(child, parent)

    # HAS_PHENOTYPE
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    for disease_id, profile in disease_profiles.items():
        for hpo_id in profile.get(key, []):
            add_edge(disease_id, hpo_id)

    return graph


# Step 2: IC weighted random walks

# Plain Node2Vec walks treat all neighbours equally
# Bias the walk toward more specific terms using IC values, so the walks
# spend more time around rare phenotypes and less around
# generic stuff like "Problem with nervous system"

def _transition_probs(
    current: str,
    previous: Optional[str],
    neighbours: List[str],
    ic_values: Dict[str, float],
    p: float,
    q: float,
) -> List[float]:
    """
    How likely we are to move to each neighbour on the next step
    IC weights push toward specific terms
    Disease nodes dont have IC so they default to a neutral weight of 1
    """
    probs = []

    for neighbour in neighbours:
        ic_weight = ic_values.get(neighbour, 1.0)  # IC weight

        if previous is None:
            # First step: wont have any bias
            bias = 1.0
        elif neighbour == previous:
            # Returning to previous node? Penalize by p
            bias = 1.0 / p
        else:
            # Check if neighbour is also a neighbour of previous
            bias = 1.0 
            # Note: full Node2Vec precomputes distance-1 sets for efficiency

        probs.append(ic_weight * bias)

    # Normalize
    total = sum(probs)
    return [p_ / total for p_ in probs]


def random_walk(
    start: str,
    graph: Dict[str, List[str]],
    ic_values: Dict[str, float],
    walk_length: int,
    p: float,
    q: float,
) -> List[str]:
    """
    Simulate one IC weighted random walk of length walk_length from start node
    Returns a sequence of node IDs
    """
    walk = [start]
    previous = None

    for _ in range(walk_length - 1):
        current = walk[-1]
        neighbours = graph.get(current, [])

        if not neighbours:
            break  #end

        probs = _transition_probs(current, previous, neighbours, ic_values, p, q)
        next_node = random.choices(neighbours, weights=probs, k=1)[0]

        previous = current
        walk.append(next_node)

    return walk


def generate_walks(
    graph: Dict[str, List[str]],
    ic_values: Dict[str, float],
    walk_length: int = WALK_LENGTH,
    walks_per_node: int = WALKS_PER_NODE,
    p: float = P,
    q: float = Q,
) -> List[List[str]]:
    """
    Generate all random walks from all nodes in the graph
    Each node gets walks_per_node walks of length walk_length
    The walks are shuffled so Word2Vec doesnt see all walks from one node together
    """
    all_nodes = list(graph.keys())
    all_walks = []

    print(f"  Generating {walks_per_node} walks × {len(all_nodes)} nodes...")

    for _ in range(walks_per_node):
        random.shuffle(all_nodes)  # shuffle
        for node in all_nodes:
            walk = random_walk(node, graph, ic_values, walk_length, p, q)
            all_walks.append(walk)

    print(f"  Total walks generated: {len(all_walks)}")
    return all_walks

# Step 3: Train Word2Vec on the walks

def train_word2vec(
    walks: List[List[str]],
    embedding_dim: int = EMBEDDING_DIM,
    window_size: int = WINDOW_SIZE,
    min_count: int = MIN_COUNT,
    workers: int = WORKERS,
    epochs: int = EPOCHS,
) -> Word2Vec:
    """
    Train Word2Vec on the random walks
    """
    print(f"  Training Word2Vec: dim={embedding_dim}, window={window_size}, epochs={epochs}...")

    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,         
    )

    print(f"  Vocabulary size: {len(model.wv)} nodes")
    return model


# Step 4 and 5: Embed patient and diseases

# Collapse a set of HPO term vectors into one vector using IC weighted averaging
# Same logic for both patient and disease

def embed_term_set(
    terms: Set[str],
    model: Word2Vec,
    ic_values: Dict[str, float],
) -> Optional[np.ndarray]:
    """
    Compute IC weighted average embedding for a set of HPO terms
    Terms not in the Word2Vec vocabulary are skipped
    Returns None if no terms have embeddings
    """
    vectors = []
    weights = []

    for term in terms:
        if term not in model.wv:
            continue
        ic = ic_values.get(term, 1.0)  # default weight 1 for terms without IC
        vectors.append(model.wv[term])
        weights.append(ic)

    if not vectors:
        return None

    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize

    # Weighted average of all term embeddings
    return np.average(vectors, axis=0, weights=weights)


#
# Step 6: Similarity and ranking


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two dense numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def rank_diseases(
    disease_profiles: Dict[str, dict],
    patient: dict,
    model: Word2Vec,
    ic_values: Dict[str, float],
    alias_to_canonical: Dict[str, str],
    use_propagated: bool = True,
    top_k: int = 10,
) -> List[dict]:
    # Build patient vector from their HPO terms
    patient_key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    patient_terms = set(patient.get(patient_key, []))
    patient_vec = embed_term_set(patient_terms, model, ic_values)

    if patient_vec is None:
        raise ValueError("No patient terms found in Word2Vec vocabulary — check your shared artifacts.")

    results = []
    disease_key = "propagated_hpo_terms" if use_propagated else "hpo_terms"

    for disease_id, profile in disease_profiles.items():
        disease_terms = set(profile.get(disease_key, []))

        if not disease_terms:
            continue

        disease_vec = embed_term_set(disease_terms, model, ic_values)
        if disease_vec is None:
            continue

        score = cosine_similarity(patient_vec, disease_vec)

        # For explanation: find patient terms that are in the disease profile
        matching_terms = [t for t in patient_terms if t in disease_terms]

        results.append({
            "disease_id": disease_id,
            "label": profile.get("label"),
            "method_name": "hpo2vec_plus",
            "score": score,
            "explanation": {
                "matching_terms": matching_terms,
                "n_matching": len(matching_terms),
                "top_ic_matches": sorted(
                    [{"term": t, "ic": ic_values.get(t, 0.0)} for t in matching_terms],
                    key=lambda x: x["ic"],
                    reverse=True,
                )[:5],
            },
            "metadata": {
                "n_patient_terms": len(patient_terms),
                "n_disease_terms": len(disease_terms),
                "used_propagated_terms": use_propagated,
            },
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    grouped = {}
    for row in results:
        canonical_id = alias_to_canonical.get(row["disease_id"], row["disease_id"])
        if canonical_id not in grouped or row["score"] > grouped[canonical_id]["score"]:
            grouped[canonical_id] = row
            grouped[canonical_id]["canonical_disease_id"] = canonical_id

    collapsed = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(collapsed, start=1):
        row["rank"] = rank

    return collapsed[:top_k]

# main

def main() -> None:

    print("No saved model found, training from scratch...")
    print("Loading shared artifacts...")
    hpo_parents = load_json(SHARED_DIR / "hpo_parents.json")
    disease_profiles = load_json(SHARED_DIR / "disease_profiles.json")
    ic_values = load_json(SHARED_DIR / "information_content.json")
    alias_to_canonical = load_json(SHARED_DIR / "alias_to_canonical.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    print(f"  {len(hpo_parents)} HPO terms, {len(disease_profiles)} disease profiles")

    model_path = HPO2VEC_DIR / "hpo2vec_model"

    print(model_path)

    if model_path.exists():
        print("Loading saved model...")
        model = Word2Vec.load(str(model_path))
    else:
        print("No saved model found, training from scratch...")

        # Step 1: Build graph
        print("Building graph...")
        graph = build_graph(hpo_parents, disease_profiles)
        print(f"  Nodes: {len(graph)}")

        # Step 2: Generate IC random walks
        print("Generating random walks...")
        walks = generate_walks(graph, ic_values)

        # Step 3: Train Word2Vec
        print("Training Word2Vec...")
        model = train_word2vec(walks)

        # Save
        model_path = HPO2VEC_DIR / "hpo2vec_model"
        model.save(str(model_path))
        print(f"  Model saved to: {model_path}")

    # Steps 4-6: Embed and rank
    print("Ranking diseases for example patient...")
    top_results = rank_diseases(
        disease_profiles=disease_profiles,
        patient=patient,
        model=model,
        ic_values=ic_values,
        alias_to_canonical=alias_to_canonical,
        use_propagated=True,
        top_k=10,
    )

    output_path = HPO2VEC_DIR / "hpo2vec_top10.json"
    save_json(top_results, output_path)

    print("\nTop HPO2Vec+ results:")
    for row in top_results:
        print(
            f"rank={row['rank']:>2} | "
            f"{row['disease_id']:<15} | "
            f"score={row['score']:.4f} | "
            f"{row['label']}"
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
