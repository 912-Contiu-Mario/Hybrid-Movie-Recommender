import pytest
import time
import threading
import logging
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import cProfile
import pstats
import io
import os
from unittest.mock import patch, MagicMock

from app.domain.models import Rating
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.recommenders.content_rec import ContentRecommender
from app.recommenders.hybrid_rec import HybridRecommender
from app.service.rec_service import RecService

# Configure logging - Set to INFO to see profiling results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force the handler to stdout to ensure it shows in test output
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False  # Prevent duplicate logging

# Shared variables for lock profiling
lock_wait_times = {}
lock_wait_counts = {}
lock_active = {}

class LockProfilingContext:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        lock_active[self.name] = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if self.name not in lock_wait_times:
            lock_wait_times[self.name] = 0
            lock_wait_counts[self.name] = 0
        lock_wait_times[self.name] += elapsed
        lock_wait_counts[self.name] += 1
        lock_active[self.name] = False

# Fixtures

@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a temporary directory with mock model files."""
    from app.models.lightgcn import LightGCN
    import torch
    import pickle
    
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Create mock model file
    model = LightGCN(num_users=100, num_items=500, embedding_dim=64, n_layers=3)
    torch.save(model.state_dict(), model_dir / "model.pt")
    
    # Create mock data file with mappings
    user_mapping = list(range(1, 101))
    item_mapping = list(range(101, 601))
    data = {
        'user_mapping': user_mapping,
        'item_mapping': item_mapping,
        'config': {
            'embedding_dim': 64,
            'n_layers': 3
        }
    }
    with open(model_dir / "lightgcn_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    return str(model_dir)

@pytest.fixture
def mock_embeddings(tmp_path):
    """Create mock content embeddings."""
    import pickle
    import numpy as np
    
    embeddings_path = tmp_path / "embeddings.pkl"
    
    # Create mock embeddings
    num_movies = 500
    embedding_dim = 384  # Standard dimension for many embedding models
    
    # Generate normalized random embeddings
    embeddings = np.random.randn(num_movies, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    movie_ids = list(range(101, 601))
    
    data = {
        'embeddings': embeddings,
        'movieId': movie_ids
    }
    
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)
    
    return str(embeddings_path)

@pytest.fixture
def test_ratings():
    """Create synthetic test ratings."""
    now = datetime.now()
    
    # Create a decent number of ratings for testing
    ratings = []
    for user_id in range(1, 50):  # 50 users
        for movie_id in range(101, 151):  # Each user rates 50 movies
            if np.random.random() < 0.6:  # 60% chance of rating each movie
                rating_value = np.random.uniform(1, 5)
                ratings.append(Rating(user_id=user_id, movie_id=movie_id, rating=rating_value, timestamp=now))
    
    logger.info(f"Created {len(ratings)} synthetic ratings for testing")
    return ratings

@pytest.fixture
def lightgcn_recommender(mock_model_dir, test_ratings):
    """Create a LightGCN recommender instance with profiling hooks."""
    
    # Reset the singleton
    LightGCNRecommender._instance = None
    
    # Patch the rwlock to add profiling
    with patch('app.recommenders.lightgcn_rec.rwlock.RWLockWrite') as mock_lock:
        # Create a mock lock that tracks timing
        class ProfilingRWLock:
            def gen_rlock(self):
                return LockProfilingContext("lightgcn_read")
            
            def gen_wlock(self):
                return LockProfilingContext("lightgcn_write")
        
        mock_lock.return_value = ProfilingRWLock()
        
        # Create the recommender
        return LightGCNRecommender.get_instance(mock_model_dir, test_ratings)

@pytest.fixture
def content_recommender(mock_embeddings):
    """Create a Content recommender instance with profiling hooks."""
    
    # Reset the singleton
    ContentRecommender._instance = None
    
    # Patch the rwlock to add profiling
    with patch('app.recommenders.content_rec.rwlock.RWLockWrite') as mock_lock:
        # Create a mock lock that tracks timing
        class ProfilingRWLock:
            def gen_rlock(self):
                return LockProfilingContext("content_read")
            
            def gen_wlock(self):
                return LockProfilingContext("content_write")
        
        mock_lock.return_value = ProfilingRWLock()
        
        # Create the recommender
        return ContentRecommender.get_instance(
            embeddings_path=mock_embeddings,
            min_rating=3.5,
            min_weight=0.1,
            max_rating=5.0
        )

@pytest.fixture
def hybrid_recommender(lightgcn_recommender, content_recommender):
    """Create a Hybrid recommender instance."""
    
    # Reset the singleton
    HybridRecommender._instance = None
    
    # Create the recommender
    return HybridRecommender.get_instance(
        lightgcn_recommender=lightgcn_recommender,
        content_recommender=content_recommender,
        alpha_max=0.9,
        alpha_min=0.5
    )

# Tests

def test_lightgcn_update_performance(lightgcn_recommender, test_ratings):
    """Test and profile the performance of LightGCN updates."""
    logger.info("=== STARTING LightGCN Update Performance Test ===")
    
    # Reset profiling data
    global lock_wait_times, lock_wait_counts
    lock_wait_times = {}
    lock_wait_counts = {}
    
    # Create a profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Perform the update
    start_time = time.time()
    lightgcn_recommender.update_final_embs(test_ratings)
    total_time = time.time() - start_time
    
    # Stop profiling
    pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    
    # Output results
    logger.info(f"LightGCN update took {total_time:.4f} seconds")
    
    # Calculate time spent in locks vs CPU
    lock_time = sum(t for name, t in lock_wait_times.items() if name.startswith('lightgcn_'))
    cpu_time = total_time - lock_time
    
    logger.info(f"Time breakdown:")
    logger.info(f"  - Total: {total_time:.4f}s")
    logger.info(f"  - Lock acquisition: {lock_time:.4f}s ({lock_time/total_time*100:.1f}%)")
    logger.info(f"  - CPU processing: {cpu_time:.4f}s ({cpu_time/total_time*100:.1f}%)")
    
    # Log lock details
    for name, time_spent in lock_wait_times.items():
        if name.startswith('lightgcn_'):
            count = lock_wait_counts[name]
            avg_time = time_spent / count if count > 0 else 0
            logger.info(f"  - {name}: {time_spent:.4f}s total, {count} acquisitions, {avg_time:.6f}s avg")
    
    # Log cProfile top 10 results (to keep output manageable)
    logger.info("cProfile top functions:")
    for line in s.getvalue().split('\n')[:15]:  # First 15 lines should include headers and top functions
        if line.strip():
            logger.info(line)
    
    # Verify the update worked
    assert lightgcn_recommender._final_user_embs is not None
    assert lightgcn_recommender._final_item_embs is not None
    
    # Determine bottleneck
    if lock_time > 0.7 * total_time:
        logger.info("CONCLUSION: Lock acquisition is the primary bottleneck")
    else:
        logger.info("CONCLUSION: CPU processing is the primary bottleneck")
    
    logger.info("=== FINISHED LightGCN Update Performance Test ===")

def test_content_recommender_performance(content_recommender, test_ratings):
    """Test content recommender recommendation performance with concurrent access."""
    logger.info("=== STARTING Content Recommender Performance Test ===")
    
    # Reset profiling data
    global lock_wait_times, lock_wait_counts
    lock_wait_times = {}
    lock_wait_counts = {}
    
    # Create a profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Extract user IDs from test ratings
    user_ids = set()
    for rating in test_ratings:
        user_ids.add(rating.user_id)
    user_ids = list(user_ids)[:5]  # Take a few users
    
    # Get user ratings
    user_ratings = {}
    for user_id in user_ids:
        user_ratings[user_id] = [r for r in test_ratings if r.user_id == user_id]
    
    # Define concurrent workload
    def recommend_for_user(user_id):
        ratings = user_ratings[user_id]
        return content_recommender.recommend(ratings, top_k=10)
    
    # Measure performance for sequential and concurrent execution
    # Sequential first
    start_time = time.time()
    for user_id in user_ids:
        recommend_for_user(user_id)
    sequential_time = time.time() - start_time
    
    # Then concurrent
    threads = []
    start_time = time.time()
    for user_id in user_ids:
        thread = threading.Thread(target=recommend_for_user, args=(user_id,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    concurrent_time = time.time() - start_time
    
    # Stop profiling
    pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    # Output results
    logger.info(f"Content recommender performance:")
    logger.info(f"  - Sequential: {sequential_time:.4f}s")
    logger.info(f"  - Concurrent: {concurrent_time:.4f}s")
    logger.info(f"  - Speedup: {sequential_time/concurrent_time:.2f}x")
    
    # Calculate lock contention
    lock_time = sum(t for name, t in lock_wait_times.items() if name.startswith('content_'))
    
    logger.info(f"Lock acquisition time: {lock_time:.4f}s")
    for name, time_spent in lock_wait_times.items():
        if name.startswith('content_'):
            count = lock_wait_counts[name]
            avg_time = time_spent / count if count > 0 else 0
            logger.info(f"  - {name}: {time_spent:.4f}s total, {count} acquisitions, {avg_time:.6f}s avg")
    
    # Log cProfile top 10 results (to keep output manageable)
    logger.info("cProfile top functions:")
    for line in s.getvalue().split('\n')[:15]:  # First 15 lines should include headers and top functions
        if line.strip():
            logger.info(line)
    
    # Check if locks are a bottleneck for concurrent execution
    if concurrent_time > 0.8 * sequential_time:
        logger.info("CONCLUSION: Limited concurrency benefit, likely due to lock contention")
    else:
        logger.info("CONCLUSION: Good concurrency benefit, locks are not a major bottleneck")
    
    logger.info("=== FINISHED Content Recommender Performance Test ===")

def test_rec_service_update_performance(mock_model_dir, mock_embeddings, test_ratings):
    """Test the performance of RecService update_ratings method."""
    logger.info("=== STARTING RecService Update Performance Test ===")
    
    # Set up mocks for repositories
    rating_repo_mock = MagicMock()
    rating_repo_mock.get_all_positive_ratings.return_value = test_ratings
    
    user_repo_mock = MagicMock()
    movie_repo_mock = MagicMock()
    
    # Config for RecService
    config = {
        'rating_threshold': 3.5,
        'update_interval_seconds': 300,
        'lightgcn': {
            'lightgcn_model_path': mock_model_dir
        },
        'content': {
            'embeddings_path': mock_embeddings,
            'min_weight': 0.1
        },
        'hybrid': {
            'alpha_max': 0.9,
            'alpha_min': 0.5
        }
    }
    
    # Reset singletons
    LightGCNRecommender._instance = None
    ContentRecommender._instance = None
    HybridRecommender._instance = None
    RecService._instance = None
    
    # Create service with profiling hooks
    with patch('app.recommenders.lightgcn_rec.rwlock.RWLockWrite') as mock_lightgcn_lock, \
         patch('app.recommenders.content_rec.rwlock.RWLockWrite') as mock_content_lock, \
         patch('app.service.rec_service.asyncio.create_task') as mock_create_task:
        
        # Create mock locks that track timing
        class ProfilingLightGCNLock:
            def gen_rlock(self):
                return LockProfilingContext("lightgcn_read")
            
            def gen_wlock(self):
                return LockProfilingContext("lightgcn_write")
        
        class ProfilingContentLock:
            def gen_rlock(self):
                return LockProfilingContext("content_read")
            
            def gen_wlock(self):
                return LockProfilingContext("content_write")
        
        mock_lightgcn_lock.return_value = ProfilingLightGCNLock()
        mock_content_lock.return_value = ProfilingContentLock()
        
        # Mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_create_task.return_value = mock_task
        
        # Reset profiling data
        global lock_wait_times, lock_wait_counts
        lock_wait_times = {}
        lock_wait_counts = {}
        
        # Create service
        rec_service = RecService(
            rating_repository=rating_repo_mock,
            user_repository=user_repo_mock,
            movie_repository=movie_repo_mock,
            config=config
        )
        
        # Create a profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Perform the update
        start_time = time.time()
        rec_service.update_ratings()
        total_time = time.time() - start_time
        
        # Stop profiling
        pr.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)
        
        # Calculate time spent in locks vs CPU
        lock_time = sum(lock_wait_times.values())
        cpu_time = total_time - lock_time
        
        # Output results
        logger.info(f"RecService update_ratings took {total_time:.4f} seconds")
        logger.info(f"Time breakdown:")
        logger.info(f"  - Total: {total_time:.4f}s")
        logger.info(f"  - Lock acquisition: {lock_time:.4f}s ({lock_time/total_time*100:.1f}%)")
        logger.info(f"  - CPU processing: {cpu_time:.4f}s ({cpu_time/total_time*100:.1f}%)")
        
        # Log lock details
        for name, time_spent in lock_wait_times.items():
            count = lock_wait_counts[name]
            avg_time = time_spent / count if count > 0 else 0
            logger.info(f"  - {name}: {time_spent:.4f}s total, {count} acquisitions, {avg_time:.6f}s avg")
        
        # Log cProfile top results
        logger.info("cProfile top functions:")
        for line in s.getvalue().split('\n')[:15]:  # First 15 lines should include headers and top functions
            if line.strip():
                logger.info(line)
        
        # Determine bottleneck
        if lock_time > 0.7 * total_time:
            logger.info("CONCLUSION: Lock acquisition is the primary bottleneck")
        else:
            logger.info("CONCLUSION: CPU processing is the primary bottleneck")
        
        logger.info("=== FINISHED RecService Update Performance Test ===")

if __name__ == "__main__":
    # Allow running the tests directly with pytest
    pytest.main(["-xvs", __file__]) 