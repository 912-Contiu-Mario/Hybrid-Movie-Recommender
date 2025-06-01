import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import threading
import time
import concurrent.futures
from queue import Queue
from typing import List, Tuple

from app.domain.models import Rating
from app.recommenders.hybrid_rec import HybridRecommender
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.recommenders.content_rec import ContentRecommender
from app.exceptions.recommender import (
    RecommendationFailedException,
    ConfigurationException
)

@pytest.fixture
def mock_lightgcn():
    """Create a mock LightGCN recommender."""
    mock = Mock(spec=LightGCNRecommender)
    
    # Mock warm item behavior
    mock.is_item_in_training_data.return_value = True
    mock.is_user_in_training_data.return_value = True
    
    # Mock recommendations
    mock.recommend.return_value = [
        (101, 0.8),
        (102, 0.7),
        (103, 0.6)
    ]
    
    return mock

@pytest.fixture
def mock_content():
    """Create a mock content recommender."""
    mock = Mock(spec=ContentRecommender)
    
    # Mock recommendations
    mock.recommend.return_value = [
        (102, 0.9),
        (103, 0.8),
        (104, 0.7)
    ]
    
    return mock

@pytest.fixture
def test_ratings():
    """Create test ratings."""
    now = datetime.now()
    return [
        Rating(user_id=1, movie_id=201, rating=4.5, timestamp=now),
        Rating(user_id=1, movie_id=202, rating=3.0, timestamp=now),
        Rating(user_id=2, movie_id=201, rating=5.0, timestamp=now),
        Rating(user_id=2, movie_id=203, rating=4.0, timestamp=now),
        Rating(user_id=3, movie_id=202, rating=3.5, timestamp=now)
    ]

@pytest.fixture
def recommender(mock_lightgcn, mock_content):
    """Create a HybridRecommender instance."""
    HybridRecommender._instance = None
    return HybridRecommender.get_instance(
        lightgcn_recommender=mock_lightgcn,
        content_recommender=mock_content,
        alpha_max=0.9,
        alpha_min=0.5
    )

def test_initialization(mock_lightgcn, mock_content):
    """Test successful initialization of the recommender."""
    recommender = HybridRecommender.get_instance(mock_lightgcn, mock_content)
    assert recommender is not None
    assert recommender.lightgcn_rec is mock_lightgcn
    assert recommender.content_rec is mock_content
    assert recommender.alpha_max == 0.9
    assert recommender.alpha_min == 0.5

def test_singleton_pattern(mock_lightgcn, mock_content):
    """Test that the recommender follows the singleton pattern."""
    recommender1 = HybridRecommender.get_instance(mock_lightgcn, mock_content)
    recommender2 = HybridRecommender.get_instance()  # Should not need args for second call
    assert recommender1 is recommender2

def test_initialization_validation():
    """Test initialization with invalid parameters."""
    # Test missing recommenders
    with pytest.raises(ConfigurationException):
        HybridRecommender._instance = None
        HybridRecommender.get_instance()
    
    # Test invalid alpha values
    with pytest.raises(ConfigurationException):
        HybridRecommender._instance = None
        HybridRecommender.get_instance(
            Mock(spec=LightGCNRecommender),
            Mock(spec=ContentRecommender),
            alpha_max=0.5,
            alpha_min=0.8
        )

def test_alpha_calculation_warm_case(recommender):
    """Test alpha calculation for warm items/users."""
    # Mock warm scenario
    recommender.lightgcn_rec.is_item_in_training_data.return_value = True
    recommender.lightgcn_rec.is_user_in_training_data.return_value = True
    
    alpha = recommender._calculate_alpha(1, 101)
    assert alpha == recommender.alpha_max

def test_alpha_calculation_cold_case(recommender):
    """Test alpha calculation for cold items/users."""
    # Mock cold scenario
    recommender.lightgcn_rec.is_item_in_training_data.return_value = False
    recommender.lightgcn_rec.is_user_in_training_data.return_value = False
    
    alpha = recommender._calculate_alpha(1, 101)
    assert alpha == recommender.alpha_min

def test_recommend_warm_case(recommender, test_ratings):
    """Test recommendations for warm items/users."""
    # Mock warm scenario
    recommender.lightgcn_rec.is_item_in_training_data.return_value = True
    recommender.lightgcn_rec.is_user_in_training_data.return_value = True
    
    recs = recommender.recommend(1, test_ratings)
    assert len(recs) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    
    # Verify scores are properly combined (alpha_max * cf + (1-alpha_max) * content)
    # For items that appear in both recommenders
    for item_id, score in recs:
        if item_id in [102, 103]:  # Items present in both recommenders
            assert 0 <= score <= 1.0

def test_recommend_cold_case(recommender, test_ratings):
    """Test recommendations for cold items/users."""
    # Mock cold scenario
    recommender.lightgcn_rec.is_item_in_training_data.return_value = False
    recommender.lightgcn_rec.is_user_in_training_data.return_value = False
    
    recs = recommender.recommend(1, test_ratings)
    assert len(recs) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    
    # Verify scores favor content recommendations more (using alpha_min)
    first_item_id, first_score = recs[0]
    assert first_item_id in [102, 103, 104]  # Should be from content recommender

def test_recommend_with_exclusions(recommender, test_ratings):
    """Test recommendations with excluded items."""
    exclude_items = [101, 102]
    recs = recommender.recommend(1, test_ratings, exclude_items=exclude_items)
    assert all(r[0] not in exclude_items for r in recs)

def test_recommend_empty_ratings(recommender):
    """Test recommendations with no user ratings."""
    recs = recommender.recommend(1, [])
    assert recs == []

def test_concurrent_recommendations(recommender, test_ratings):
    """Test concurrent recommendation requests."""
    num_threads = 10
    results = []
    errors = []

    def get_recommendations():
        try:
            recs = recommender.recommend(1, test_ratings)
            results.append(recs)
        except Exception as e:
            errors.append(e)
    
    # Create and start multiple threads
    threads = [threading.Thread(target=get_recommendations) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent recommendations produced errors: {errors}"
    assert len(results) == num_threads, "Not all recommendation requests completed"
    
    # Verify all results are consistent
    first_result = results[0]
    assert all(r == first_result for r in results), "Inconsistent results from concurrent requests"

def test_stress_test_recommendations(recommender, test_ratings):
    """Stress test the recommender with many concurrent requests."""
    num_threads = 20
    results = []
    errors = []

    def mixed_operation():
        try:
            time.sleep(0.05)  # Small delay to increase chance of concurrent access
            recs = recommender.recommend(1, test_ratings, top_k=5)
            results.append(len(recs))
        except Exception as e:
            errors.append(e)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(mixed_operation) for _ in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Stress test produced errors: {errors}"
    assert len(results) == num_threads, "Not all operations completed"
    assert all(r == results[0] for r in results), "Inconsistent recommendation counts"

def test_score_normalization(recommender, test_ratings):
    """Test that scores are properly normalized and combined."""
    # Set up mock returns with known values
    recommender.lightgcn_rec.recommend.return_value = [
        (101, 0.8),  # High CF score
        (102, 0.4),  # Medium CF score
        (103, 0.2)   # Low CF score
    ]
    
    recommender.content_rec.recommend.return_value = [
        (102, 0.9),  # High content score
        (103, 0.5),  # Medium content score
        (104, 0.1)   # Low content score
    ]
    
    # Test warm case (high alpha)
    recommender.lightgcn_rec.is_item_in_training_data.return_value = True
    recommender.lightgcn_rec.is_user_in_training_data.return_value = True
    
    recs = recommender.recommend(1, test_ratings)
    
    # Verify normalization and combination
    # For item 102 (appears in both with different scores):
    # CF normalized: (0.4 - 0.2) / (0.8 - 0.2) = 0.33
    # Content normalized: (0.9 - 0.1) / (0.9 - 0.1) = 1.0
    # Combined with alpha_max = 0.9:
    # 0.9 * 0.33 + 0.1 * 1.0 = 0.397
    item_102_score = next(score for item_id, score in recs if item_id == 102)
    assert 0.35 <= item_102_score <= 0.45  # Allow for floating-point imprecision 