"""HPO2Vec similarity methods"""

import random
from typing import Dict, List, Optional, Set

import numpy as np
from gensim.models import Word2Vec

# Parameters
# Controls the random walk and Word2Vec training behaviour
# My defaults

WALK_LENGTH = 80  # number of steps per random walk
WALKS_PER_NODE = 10  # how many walks to start from each node
P = 1.0  # return param: controls likelihood of revisiting a node
# lower p = more likely to backtrack
Q = 0.5  # in out parameter: controls BFS vs DFS behaviour
# q < 1 = biased toward DFS
# q > 1 = toward BFS
EMBEDDING_DIM = 128  # dimension of the learned embeddings
WINDOW_SIZE = 10  # Word2Vec context window aka how many neighbours to consider
MIN_COUNT = 1  # minimum term frequency to include in vocabulary
WORKERS = 4  # parallel workers for Word2Vec training
EPOCHS = 5  # training epochs


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
    terms_key: str = "hpo_terms",
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
    for disease_id, profile in disease_profiles.items():
        for hpo_id in profile.get(terms_key, []):
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
        ic_weight = ic_values.get(neighbour, 1.0)

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
            break  # end

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
    print(
        f"  Training Word2Vec: dim={embedding_dim}, window={window_size}, epochs={epochs}..."
    )

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

    return np.average(vectors, axis=0, weights=weights)


#
# Step 6: Similarity and ranking


def cosine_similarity_np(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two dense numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
