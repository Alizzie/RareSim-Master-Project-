"""Configuration file for the hpo2vec similarity method."""

from raresim.utils.paths import SIMILARITY_DIR, MODELS_DIR

# --- Random Walk Parameters ---

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

# --- Method Constants ---
PIPELINE_NAME = "hpo2vec"
HPO2VEC_DIR = SIMILARITY_DIR / PIPELINE_NAME
HPO2VECPLUS = "hpo2vec_plus"
ALL_METHOD = [HPO2VECPLUS]

MODEL_CACHE_DIR = HPO2VEC_DIR / "model_cache"
