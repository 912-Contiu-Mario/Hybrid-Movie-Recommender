from typing import Dict, Any
from app.config.paths import LIGHTGCN_MODEL_DATA_DIR, COMBINED_EMBEDDINGS_PATH

# Global rating threshold used across all recommenders
# This threshold MUST match the one used during LightGCN training
# to ensure consistency between collaborative and content-based recommendations
RATING_THRESHOLD = 3.5

# Content-based recommender settings
CONTENT_MIN_WEIGHT = 0.7  # Weight that 3.5 rating gets over 5 rating
CONTENT_MAX_WEIGHT = 1.0  # Weight that 5.0 rating gets

# Hybrid recommender settings
ALPHA_MAX = 0.9  # Maximum weight for collaborative filtering in hybrid recommendations
ALPHA_MIN = 0.2 # Minimum weight for collaborative filtering in hybrid recommendations

# periodic update interval in seconds
UPDATE_INTERVAL_SECONDS = 60

def get_recommender_config() -> Dict[str, Any]:
    return {
        'rating_threshold': RATING_THRESHOLD,
        'content': {
            'embeddings_path': COMBINED_EMBEDDINGS_PATH,
            'min_weight': CONTENT_MIN_WEIGHT,
            'max_weight': CONTENT_MAX_WEIGHT
        },
        'hybrid': {
            'alpha_max': ALPHA_MAX,
            'alpha_min': ALPHA_MIN
        },
        'lightgcn': {
            'lightgcn_model_path': LIGHTGCN_MODEL_DATA_DIR,
        },
        'update_interval_seconds': UPDATE_INTERVAL_SECONDS
    } 
