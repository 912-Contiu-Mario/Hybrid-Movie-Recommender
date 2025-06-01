from typing import Dict, Any

from app.config import paths

def get_training_default_config() -> Dict[str, Any]:
    """Get default configuration for LightGCN training."""
    return {
        'embedding_dim': 64,
        'n_layers': 2,
        'add_self_loops': False,
        'lambda_val': 0.0001,
        'lr': 0.001,
        'batch_size': 2048,
        'iterations': 30000,
        'eval_every': 200,
        'k': 20,
        'user_min_interactions': 0,
        'item_min_interactions': 0,
        'user_max_interactions': 0,
        'item_max_interactions': 0,
        'val_size': 0.1,
        'test_size': 0.2,
        'rating_threshold': 3.5,
        'random_seed': 42,
        'models_dir': paths.LIGHTGCN_DIR,
        # Service configuration
        'api_url': 'http://localhost:8000',
        'training_schedule': {
            'interval_hours': 24,
            'start_hour': 2,
            'enabled': True
        }
    }

def load_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration with optional custom overrides."""
    config = get_training_default_config()
    if custom_config is not None:
        config.update(custom_config)
    return config

def get_training_config() -> Dict[str, Any]:
    """Get the training configuration."""
    return get_training_default_config()

