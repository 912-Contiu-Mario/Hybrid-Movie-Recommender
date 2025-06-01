import pytest
import os
import numpy as np
import pickle
import faiss
from datetime import datetime
import threading
import time
import concurrent.futures
from queue import Queue
from typing import List, Tuple

from app.domain.models import Rating
from app.recommenders.content_rec import ContentRecommender
from app.exceptions.recommender import (
    ModelNotLoadedException,
    RecommenderException,
    ResourceNotFoundException,
    InvalidRequestException,
    ConfigurationException
)

@pytest.fixture
def mock_embeddings_dir(tmp_path):
    """Create a temporary directory with mock embeddings."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    
    # Create mock embeddings file
    num_movies = 5
    embedding_dim = 8
    
    # Create normalized embeddings (for cosine similarity)
    embeddings = np.random.randn(num_movies, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    movie_ids = [101, 102, 103, 104, 105]
    
    data = {
        'embeddings': embeddings.astype(np.float32),
        'movieId': movie_ids
    }
    
    embeddings_path = embeddings_dir / "embeddings.pkl"
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)
    
    return str(embeddings_path)

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
def recommender(mock_embeddings_dir):
    """Create a ContentRecommender instance."""
    ContentRecommender._instance = None
    return ContentRecommender.get_instance(
        embeddings_path=mock_embeddings_dir,
        min_rating=3.5,
        min_weight=0.7,
        max_rating=5.0
    )

def test_initialization(mock_embeddings_dir):
    """Test successful initialization of the recommender."""
    recommender = ContentRecommender.get_instance(mock_embeddings_dir)
    assert recommender is not None
    assert recommender._embeddings is not None
    assert recommender._movie_ids is not None
    assert recommender._movie_idx_map is not None
    assert recommender._faiss_index is not None

def test_singleton_pattern(mock_embeddings_dir):
    """Test that the recommender follows the singleton pattern."""
    recommender1 = ContentRecommender.get_instance(mock_embeddings_dir)
    recommender2 = ContentRecommender.get_instance()  # Should not need args for second call
    assert recommender1 is recommender2

def test_get_item_embedding(recommender):
    """Test retrieving item embeddings."""
    # Test existing item
    emb = recommender.get_item_embedding(101)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)  # embedding_dim from fixture
    
    # Test non-existent item
    with pytest.raises(ResourceNotFoundException):
        recommender.get_item_embedding(999)

def test_get_item_similarity(recommender):
    """Test similarity calculation between items."""
    # Test similarity between existing items
    sim = recommender.get_item_similarity(101, 102)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0  # Cosine similarity range
    
    # Test with non-existent item
    assert recommender.get_item_similarity(101, 999) is None

def test_get_content_score(recommender, test_ratings):
    """Test content score calculation."""
    # Test score for existing item and ratings
    score = recommender.get_content_score(103, test_ratings)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    
    # Test with empty ratings
    assert recommender.get_content_score(101, []) is None

def test_recommend(recommender, test_ratings):
    """Test recommendation generation."""
    # Test basic recommendation
    recs = recommender.recommend(test_ratings)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    
    # Test with exclusions
    exclude_items = [101, 102]
    recs = recommender.recommend(test_ratings, exclude_items=exclude_items)
    assert all(r[0] not in exclude_items for r in recs)
    
    # Test with custom top_k
    recs = recommender.recommend(test_ratings, top_k=2)
    assert len(recs) == 2

def test_verify_concurrent_execution(recommender, test_ratings, mock_embeddings_dir):
    """Test that operations can truly run concurrently by verifying timing overlaps."""
    operation_log = Queue()
    
    def log_operation(op_name: str, start: bool):
        timestamp = time.time()
        operation_log.put((timestamp, op_name, "start" if start else "end"))
    
    def long_running_update():
        log_operation("update", True)
        time.sleep(0.5)  # Simulate a long update
        recommender.update_embs(mock_embeddings_dir)
        log_operation("update", False)
    
    def get_recommendations():
        log_operation("recommend", True)
        result = recommender.recommend(test_ratings, top_k=2)
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

def test_concurrent_reads(recommender, test_ratings):
    """Test concurrent read operations are thread-safe."""
    num_threads = 10
    results = []
    errors = []

    def read_operation():
        try:
            # Mix of different read operations
            recommender.get_item_embedding(101)
            recommender.get_item_similarity(101, 102)
            recommender.get_content_score(103, test_ratings)
            recommender.recommend(test_ratings, top_k=2)
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

def test_concurrent_embeddings_update(recommender, test_ratings, mock_embeddings_dir):
    """Test concurrent embeddings updates and reads."""
    num_readers = 5
    num_writers = 3
    read_results = []
    write_results = []
    errors = []

    def read_operation():
        try:
            time.sleep(0.1)  # Simulate some work
            recommender.get_item_embedding(101)
            recommender.get_content_score(103, test_ratings)
            read_results.append(True)
        except Exception as e:
            errors.append(e)

    def write_operation():
        try:
            time.sleep(0.1)  # Simulate some work
            recommender.update_embs(mock_embeddings_dir)
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

def test_stress_test(recommender, test_ratings, mock_embeddings_dir):
    """Stress test the recommender with mixed operations."""
    num_threads = 20
    results = []
    errors = []

    def mixed_operation(op_type):
        try:
            time.sleep(0.05)  # Small delay to increase chance of concurrent access
            if op_type == 'read':
                recommender.get_content_score(101, test_ratings)
                recommender.recommend(test_ratings, top_k=2)
            elif op_type == 'write':
                recommender.update_embs(mock_embeddings_dir)
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