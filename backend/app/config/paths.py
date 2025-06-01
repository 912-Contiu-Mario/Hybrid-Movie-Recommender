from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent


DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed_movie_lens"
ENCODINGS_DIR = DATA_DIR / "encodings"
LIGHTGCN_DIR = DATA_DIR / "lightgcn" / "saved_models"

# Data files
MOVIES_PATH = PROCESSED_DIR / "movies.csv"
RATINGS_SMALL_PATH = PROCESSED_DIR / "ratings.csv"
RATINGS_FULL_PATH = PROCESSED_DIR / "full_ratings.csv"

# model and embeddings paths
LIGHTGCN_MODEL_DATA_DIR = LIGHTGCN_DIR / "latest-01-val-02-train"
COMBINED_EMBEDDINGS_PATH = ENCODINGS_DIR / "mpnet_genre_language_combined_embeddings.pkl"
