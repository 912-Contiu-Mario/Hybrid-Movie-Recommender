from app.config.paths import *
from app.config.recommender import *

VERSION = "0.1.0"
API_TITLE = "Movie Recommendation API"
API_DESCRIPTION = "API for hybrid movie recommendations"


def validate_config():
    if not RATING_THRESHOLD:
        raise ValueError("RATING_THRESHOLD must be set")
    if ALPHA_MIN >= ALPHA_MAX:
        raise ValueError("ALPHA_MIN must be less than ALPHA_MAX")
    if not all(Path(p).exists() for p in [MOVIES_PATH, RATINGS_SMALL_PATH]):
        raise FileNotFoundError("Required data files not found")
    if not LIGHTGCN_MODEL_DATA_DIR.exists():
        raise ValueError("LIGHTGCN_MODEL_DATA_DIR must be set")
    if not COMBINED_EMBEDDINGS_PATH.exists():
        raise ValueError("COMBINED_EMBEDDINGS_PATH must be set")


validate_config()