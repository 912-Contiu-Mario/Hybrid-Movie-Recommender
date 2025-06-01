import pytest
import os
import torch
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import threading
import time
import concurrent.futures
from queue import Queue
from typing import List, Tuple

from app.domain.models import Rating
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.models.lightgcn import LightGCN
from app.exceptions.recommender import (
    ModelNotLoadedException,
    RecommenderException,
    ResourceNotFoundException,
    InvalidRequestException,
    ConfigurationException
)

@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a temporary directory with mock model files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Create mock model file
    model = LightGCN(num_users=3, num_items=4, embedding_dim=8, n_layers=2)
    torch.save(model.state_dict(), model_dir / "model.pt")
    
    # Create mock data file with mappings
    import pickle
    data = {
        'user_mapping': [1, 2, 3],  # user IDs
        'item_mapping': [101, 102, 103, 104],  # movie IDs
        'config': {
            'embedding_dim': 8,
            'n_layers': 2
        }
    }
    with open(model_dir / "lightgcn_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    return str(model_dir)

@pytest.fixture
def test_ratings():
    """Create test ratings."""
    now = datetime.now()
    return [
        Rating(user_id=1, movie_id=101, rating=4.5, timestamp=now),
        Rating(user_id=1, movie_id=102, rating=3.0, timestamp=now),
        Rating(user_id=2, movie_id=101, rating=5.0, timestamp=now),
        Rating(user_id=2, movie_id=103, rating=4.0, timestamp=now),
        Rating(user_id=3, movie_id=102, rating=3.5, timestamp=now)
    ]

@pytest.fixture
def recommender(mock_model_dir, test_ratings):
    """Create a LightGCN recommender instance."""
    # Reset the singleton instance
    LightGCNRecommender._instance = None
    return LightGCNRecommender.get_instance(mock_model_dir, test_ratings)

def test_initialization(mock_model_dir, test_ratings):
    """Test successful initialization of the recommender."""
    recommender = LightGCNRecommender.get_instance(mock_model_dir, test_ratings)
    assert recommender is not None
    assert recommender._model is not None
    assert recommender._final_user_embs is not None
    assert recommender._final_item_embs is not None

def test_singleton_pattern(mock_model_dir, test_ratings):
    """Test that the recommender follows the singleton pattern."""
    recommender1 = LightGCNRecommender.get_instance(mock_model_dir, test_ratings)
    recommender2 = LightGCNRecommender.get_instance()  # Should not need args for second call
    assert recommender1 is recommender2

def test_initialization_without_ratings(mock_model_dir):
    """Test that initialization fails without ratings."""
    with pytest.raises(ConfigurationException, match="ratings must be provided for first initialization"):
        LightGCNRecommender._instance = None
        LightGCNRecommender.get_instance(mock_model_dir, None)

def test_initialization_without_model_dir():
    """Test that initialization fails without model directory."""
    with pytest.raises(ConfigurationException, match="model_dir must be provided for first initialization"):
        LightGCNRecommender._instance = None
        LightGCNRecommender.get_instance(None, [Rating(user_id=1, movie_id=101, rating=4.5, timestamp=datetime.now())])

def test_get_user_embedding(recommender):
    """Test retrieving user embeddings."""
    # Test existing user
    emb = recommender.get_user_embedding(1)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)  # embedding_dim from config
    
    # Test non-existent user
    with pytest.raises(ResourceNotFoundException):
        recommender.get_user_embedding(999)

def test_get_item_embedding(recommender):
    """Test retrieving item embeddings."""
    # Test existing item
    emb = recommender.get_item_embedding(101)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)  # embedding_dim from config
    
    # Test non-existent item
    with pytest.raises(ResourceNotFoundException):
        recommender.get_item_embedding(999)

def test_get_collab_score(recommender):
    """Test collaborative filtering score calculation."""
    # Test score for existing user-item pair
    score = recommender.get_collab_score(1, 101)
    assert isinstance(score, float)
    
    # Test score for non-existent user
    score = recommender.get_collab_score(999, 101)
    assert score is None
    
    # Test score for non-existent item
    score = recommender.get_collab_score(1, 999)
    assert score is None

def test_recommend(recommender):
    """Test recommendation generation."""
    # Test basic recommendation
    recs = recommender.recommend(1)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    assert all(isinstance(r[0], int) and isinstance(r[1], float) for r in recs)
    
    # Test with exclusions
    exclude_items = [101, 102]
    recs = recommender.recommend(1, exclude_items=exclude_items)
    assert all(r[0] not in exclude_items for r in recs)
    
    # Test with custom top_k
    recs = recommender.recommend(1, top_k=2)
    assert len(recs) == 2
    
    # Test with non-existent user
    recs = recommender.recommend(999)
    assert len(recs) == 0

def test_is_user_in_training_data(recommender):
    """Test checking if a user is in training data."""
    assert recommender.is_user_in_training_data(1) is True
    assert recommender.is_user_in_training_data(999) is False

def test_is_item_in_training_data(recommender):
    """Test checking if an item is in training data."""
    assert recommender.is_item_in_training_data(101) is True
    assert recommender.is_item_in_training_data(999) is False

def test_update_final_embs(recommender, test_ratings):
    """Test updating embeddings with new ratings."""
    # Get original embeddings
    original_user_emb = recommender.get_user_embedding(1).copy()
    
    # Update with new ratings
    new_ratings = test_ratings + [
        Rating(user_id=1, movie_id=104, rating=5.0, timestamp=datetime.now())
    ]
    recommender.update_final_embs(new_ratings)
    
    # Check that embeddings have been updated
    new_user_emb = recommender.get_user_embedding(1)
    assert not np.array_equal(original_user_emb, new_user_emb)

def test_invalid_device_configuration():
    """Test initialization with invalid device."""
    with pytest.raises(ConfigurationException):
        LightGCNRecommender._instance = None
        LightGCNRecommender.get_instance("some_dir", [], device="invalid_device")

def test_concurrent_reads(recommender):
    """Test concurrent read operations are thread-safe."""
    num_threads = 10
    results = []
    errors = []

    def read_operation():
        try:
            # Mix of different read operations
            recommender.get_user_embedding(1)
            recommender.get_item_embedding(101)
            recommender.get_collab_score(1, 101)
            recommender.recommend(1, top_k=2)
            results.append(True)
        except Exception as e:
            errors.append(e)
    
    # Create and start multiple threads
    threads = [threading.Thread(target=read_operation) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent reads produced errors: {errors}"
    assert len(results) == num_threads, "Not all read operations completed"

def test_concurrent_read_write(recommender, test_ratings):
    """Test concurrent read and write operations are thread-safe."""
    num_readers = 5
    num_writers = 3
    read_results = []
    write_results = []
    errors = []

    def read_operation():
        try:
            time.sleep(0.1)  # Simulate some work
            recommender.get_user_embedding(1)
            recommender.get_collab_score(1, 101)
            read_results.append(True)
        except Exception as e:
            errors.append(e)

    def write_operation():
        try:
            time.sleep(0.1)  # Simulate some work
            new_ratings = test_ratings + [
                Rating(user_id=1, movie_id=104, rating=5.0, timestamp=datetime.now())
            ]
            recommender.update_final_embs(new_ratings)
            write_results.append(True)
        except Exception as e:
            errors.append(e)

    # Create mix of reader and writer threads
    threads = (
        [threading.Thread(target=read_operation) for _ in range(num_readers)] +
        [threading.Thread(target=write_operation) for _ in range(num_writers)]
    )

    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent operations produced errors: {errors}"
    assert len(read_results) == num_readers, "Not all read operations completed"
    assert len(write_results) == num_writers, "Not all write operations completed"

def test_concurrent_model_updates(recommender, test_ratings, mock_model_dir):
    """Test concurrent model updates are thread-safe."""
    num_threads = 5
    update_results = []
    errors = []

    def update_operation():
        try:
            time.sleep(0.1)  # Simulate some work
            # Try to load the same model again (should be safe)
            recommender.load_new_model(mock_model_dir, test_ratings)
            update_results.append(True)
        except Exception as e:
            errors.append(e)

    # Use ThreadPoolExecutor to manage concurrent updates
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(update_operation) for _ in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent model updates produced errors: {errors}"
    assert len(update_results) == num_threads, "Not all update operations completed"

def test_stress_test(recommender, test_ratings):
    """Stress test the recommender with mixed operations."""
    num_threads = 20
    results = []
    errors = []

    def mixed_operation(op_type):
        try:
            time.sleep(0.05)  # Small delay to increase chance of concurrent access
            if op_type == 'read':
                recommender.get_collab_score(1, 101)
                recommender.recommend(1, top_k=2)
            elif op_type == 'write':
                new_ratings = test_ratings + [
                    Rating(user_id=1, movie_id=104, rating=float(np.random.randint(1, 6)), timestamp=datetime.now())
                ]
                recommender.update_final_embs(new_ratings)
            results.append(True)
        except Exception as e:
            errors.append(e)

    # Create mix of operations (70% reads, 30% writes)
    operations = ['read'] * 14 + ['write'] * 6
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(mixed_operation, op) for op in operations]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Stress test produced errors: {errors}"
    assert len(results) == num_threads, "Not all operations completed"

def test_verify_concurrent_execution(recommender, test_ratings):
    """Test that operations can truly run concurrently by verifying timing overlaps."""
    operation_log = Queue()
    
    def log_operation(op_name: str, start: bool):
        timestamp = time.time()
        operation_log.put((timestamp, op_name, "start" if start else "end"))
    
    def long_running_update():
        log_operation("update", True)
        time.sleep(0.5)  # Simulate a long update
        new_ratings = test_ratings + [
            Rating(user_id=1, movie_id=104, rating=5.0, timestamp=datetime.now())
        ]
        recommender.update_final_embs(new_ratings)
        log_operation("update", False)
    
    def get_recommendations():
        log_operation("recommend", True)
        result = recommender.recommend(1, top_k=2)
        log_operation("recommend", False)
        return result
    
    # Start the long update in a separate thread
    update_thread = threading.Thread(target=long_running_update)
    update_thread.start()
    
    # Wait a bit to ensure the update has started
    time.sleep(0.1)
    
    # Try to get recommendations while update is running
    recommendations = get_recommendations()
    
    # Wait for update to complete
    update_thread.join()
    
    # Convert operation log to list for analysis
    operations: List[Tuple[float, str, str]] = []
    while not operation_log.empty():
        operations.append(operation_log.get())
    
    # Sort operations by timestamp
    operations.sort(key=lambda x: x[0])
    
    # Find the operation intervals
    update_start = next(op[0] for op in operations if op[1] == "update" and op[2] == "start")
    update_end = next(op[0] for op in operations if op[1] == "update" and op[2] == "end")
    recommend_start = next(op[0] for op in operations if op[1] == "recommend" and op[2] == "start")
    recommend_end = next(op[0] for op in operations if op[1] == "recommend" and op[2] == "end")
    
    # Verify operations overlapped in time
    assert recommend_start > update_start and recommend_start < update_end, \
        "Recommendation did not start during the update operation"
    assert len(recommendations) > 0, \
        "Recommendations should be available even during update"

def test_concurrent_model_load_and_recommend(recommender, test_ratings, mock_model_dir):
    """Test that recommendations work while model is being reloaded."""
    operation_log = Queue()
    errors = []
    recommendations = []
    
    def log_operation(op_name: str, start: bool):
        timestamp = time.time()
        operation_log.put((timestamp, op_name, "start" if start else "end"))
    
    def slow_model_load():
        try:
            log_operation("model_load", True)
            time.sleep(0.5)  # Simulate slow model loading
            recommender.load_new_model(mock_model_dir, test_ratings)
            log_operation("model_load", False)
        except Exception as e:
            errors.append(e)
    
    def get_recommendations():
        try:
            log_operation("recommend", True)
            result = recommender.recommend(1, top_k=2)
            recommendations.append(result)
            log_operation("recommend", False)
        except Exception as e:
            errors.append(e)
    
    # Start multiple recommendation requests while loading model
    load_thread = threading.Thread(target=slow_model_load)
    recommend_threads = [
        threading.Thread(target=get_recommendations)
        for _ in range(3)
    ]
    
    load_thread.start()
    time.sleep(0.1)  # Ensure load has started
    
    for t in recommend_threads:
        t.start()
    
    # Wait for all operations to complete
    load_thread.join()
    for t in recommend_threads:
        t.join()
    
    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent operations produced errors: {errors}"
    
    # Verify all recommendation requests got results
    assert len(recommendations) == 3, "Not all recommendation requests completed"
    assert all(len(rec) > 0 for rec in recommendations), "Some recommendations were empty"
    
    # Analyze operation log
    operations = []
    while not operation_log.empty():
        operations.append(operation_log.get())
    operations.sort(key=lambda x: x[0])
    
    # Verify at least one recommendation overlapped with model loading
    model_load_start = next(op[0] for op in operations if op[1] == "model_load" and op[2] == "start")
    model_load_end = next(op[0] for op in operations if op[1] == "model_load" and op[2] == "end")
    
    recommend_times = [
        (op[0], op[2]) for op in operations if op[1] == "recommend"
    ]
    
    overlapping_recommends = [
        start for start, state in recommend_times if state == "start"
        and model_load_start < start < model_load_end
    ]
    
    assert len(overlapping_recommends) > 0, \
        "No recommendations were processed during model loading" 