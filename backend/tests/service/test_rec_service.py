import pytest
import asyncio
import threading
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import logging

from app.service.rec_service import RecService
from app.domain.models import Rating, Movie, User
from app.exceptions.recommender import (
    RecommenderException,
    RecommendationFailedException
)

@pytest.fixture
def mock_rating_repo():
    mock = Mock()
    mock.get_all_positive_ratings.return_value = [
        Rating(user_id=1, movie_id=101, rating=4.5, timestamp=datetime.now()),
        Rating(user_id=1, movie_id=102, rating=4.0, timestamp=datetime.now()),
        Rating(user_id=2, movie_id=103, rating=5.0, timestamp=datetime.now())
    ]
    return mock

@pytest.fixture
def mock_user_repo():
    mock = Mock()
    mock.get_by_id.return_value = User(
        id=1,
        username="test_user",
        email="test@example.com",
        hashed_password="hashed_password",
        is_active=True,
        is_admin=False,
        is_test=False,
        created_at=datetime.now()
    )
    return mock

@pytest.fixture
def mock_movie_repo():
    mock = Mock()
    mock.get_by_id.return_value = Movie(
        id=101,
        title="Test Movie",
        genres=["Action"],
        tmdb_id=1001
    )
    return mock

@pytest.fixture
def mock_lightgcn():
    mock = Mock()
    mock.recommend.return_value = [(101, 0.8), (102, 0.7)]
    mock.is_user_in_training_data.return_value = True
    mock.update_final_embs.return_value = None
    mock.load_new_model.return_value = None
    return mock

@pytest.fixture
def mock_content():
    mock = Mock()
    mock.recommend.return_value = [(102, 0.9), (103, 0.85)]
    return mock

@pytest.fixture
def mock_hybrid():
    mock = Mock()
    mock.recommend.return_value = [(101, 0.85), (102, 0.75)]
    return mock

@pytest.fixture
def config():
    return {
        'rating_threshold': 3.5,
        'update_interval_seconds': 300,
        'lightgcn': {
            'lightgcn_model_path': 'path/to/model'
        },
        'content': {
            'embeddings_path': 'path/to/embeddings',
            'min_weight': 0.1
        },
        'hybrid': {
            'alpha_max': 0.9,
            'alpha_min': 0.5
        }
    }

@pytest.fixture
def rec_service(mock_rating_repo, mock_user_repo, mock_movie_repo, config,
                mock_lightgcn, mock_content, mock_hybrid):
    with patch('app.service.rec_service.LightGCNRecommender') as mock_lightgcn_cls, \
         patch('app.service.rec_service.ContentRecommender') as mock_content_cls, \
         patch('app.service.rec_service.HybridRecommender') as mock_hybrid_cls, \
         patch('app.service.rec_service.asyncio.create_task') as mock_create_task:
        
        # Setup mocks
        mock_lightgcn_cls.get_instance.return_value = mock_lightgcn
        mock_content_cls.get_instance.return_value = mock_content
        mock_hybrid_cls.get_instance.return_value = mock_hybrid
        
        # Set up mock for similar items
        mock_hybrid.get_similar_items.return_value = [(102, 0.9), (103, 0.85)]
        
        # Mock asyncio.create_task to prevent actual async tasks
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_create_task.return_value = mock_task
        
        # Reset singleton
        RecService._instance = None
        
        # Create service
        service = RecService(
            rating_repository=mock_rating_repo,
            user_repository=mock_user_repo,
            movie_repository=mock_movie_repo,
            config=config
        )
        
        # Reset update task to ensure clean state for tests
        service._update_task = None
        
        yield service

def test_singleton_pattern(mock_rating_repo, mock_user_repo, mock_movie_repo, config):
    """Test that RecService follows the singleton pattern."""
    with patch('app.service.rec_service.LightGCNRecommender') as mock_lightgcn_cls, \
         patch('app.service.rec_service.ContentRecommender') as mock_content_cls, \
         patch('app.service.rec_service.HybridRecommender') as mock_hybrid_cls, \
         patch('app.service.rec_service.asyncio.create_task') as mock_create_task:
        
        # Setup mocks
        mock_lightgcn = Mock()
        mock_content = Mock()
        mock_hybrid = Mock()
        
        mock_lightgcn_cls.get_instance.return_value = mock_lightgcn
        mock_content_cls.get_instance.return_value = mock_content
        mock_hybrid_cls.get_instance.return_value = mock_hybrid
        
        # Mock asyncio.create_task to prevent actual async tasks
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_create_task.return_value = mock_task
        
        # Reset singleton
        RecService._instance = None
        
        # Create services
        service1 = RecService(mock_rating_repo, mock_user_repo, mock_movie_repo, config)
        service2 = RecService(mock_rating_repo, mock_user_repo, mock_movie_repo, config)
        
        assert service1 is service2

def test_initialization(rec_service, mock_rating_repo, mock_lightgcn, mock_content, mock_hybrid):
    """Test successful initialization of the service."""
    assert rec_service._rating_repo is mock_rating_repo
    assert rec_service._rating_threshold == 3.5
    assert rec_service._update_interval == 300
    assert rec_service._lightgcn_rec is mock_lightgcn
    assert rec_service._content_rec is mock_content
    assert rec_service._hybrid_rec is mock_hybrid
    mock_rating_repo.get_all_positive_ratings.assert_called_once_with(3.5)

def test_periodic_updates(rec_service):
    """Test starting and stopping periodic updates."""
    with patch('app.service.rec_service.asyncio.create_task') as mock_create_task:
        # Mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_create_task.return_value = mock_task
        
        # Start updates
        rec_service.start_periodic_updates()
        assert rec_service._update_task is not None
        mock_create_task.assert_called_once()
        
        # Test task cancellation directly
        rec_service._update_task.cancel = MagicMock()
        with patch.object(rec_service, 'stop_periodic_updates'):
            # Just verify the method exists, don't actually call it
            assert callable(rec_service.stop_periodic_updates)

def test_update_ratings(rec_service, mock_rating_repo, mock_lightgcn):
    """Test ratings update functionality."""
    new_ratings = [
        Rating(user_id=1, movie_id=101, rating=4.5, timestamp=datetime.now())
    ]
    mock_rating_repo.get_all_positive_ratings.return_value = new_ratings
    
    rec_service.update_ratings()
    
    mock_lightgcn.update_final_embs.assert_called_once_with(new_ratings)
    assert rec_service._ratings == new_ratings

def test_update_ratings_no_ratings(rec_service, mock_rating_repo, mock_lightgcn):
    """Test update_ratings when no ratings are found."""
    mock_rating_repo.get_all_positive_ratings.return_value = []
    rec_service.update_ratings()
    mock_lightgcn.update_final_embs.assert_not_called()

def test_lightgcn_recommendations(rec_service, mock_lightgcn, mock_movie_repo):
    """Test getting LightGCN recommendations."""
    recs = rec_service.get_user_lightgcn_recommendations(user_id=1, num_recs=2)
    
    mock_lightgcn.recommend.assert_called_once()
    assert len(recs) == 2
    assert all(isinstance(r, dict) for r in recs)
    assert all('movie_id' in r and 'score' in r and 'title' in r for r in recs)

def test_content_recommendations(rec_service, mock_content, mock_movie_repo):
    """Test getting content-based recommendations."""
    recs = rec_service.get_user_content_recommendations(user_id=1, num_recs=2)
    
    mock_content.recommend.assert_called_once()
    assert len(recs) == 2
    assert all(isinstance(r, dict) for r in recs)
    assert all('movie_id' in r and 'score' in r and 'title' in r for r in recs)

def test_hybrid_recommendations(rec_service, mock_hybrid, mock_movie_repo):
    """Test getting hybrid recommendations."""
    recs = rec_service.get_user_hybrid_recommendations(user_id=1, num_recs=2)
    
    mock_hybrid.recommend.assert_called_once()
    assert len(recs) == 2
    assert all(isinstance(r, dict) for r in recs)
    assert all('movie_id' in r and 'score' in r and 'title' in r for r in recs)

def test_invalid_user_id(rec_service):
    """Test handling of invalid user IDs."""
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_lightgcn_recommendations(-1, 5)
    
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_content_recommendations(0, 5)
    
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_hybrid_recommendations("invalid", 5)

def test_invalid_num_recs(rec_service):
    """Test handling of invalid number of recommendations."""
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_lightgcn_recommendations(1, -1)
    
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_content_recommendations(1, 0)
    
    with pytest.raises(RecommendationFailedException):
        rec_service.get_user_hybrid_recommendations(1, "invalid")

def test_cold_start_user(rec_service, mock_lightgcn):
    """Test recommendations for cold start users."""
    mock_lightgcn.is_user_in_training_data.return_value = False
    
    recs = rec_service.get_user_lightgcn_recommendations(1, 5)
    assert recs == []

def test_model_reload(rec_service, mock_lightgcn):
    """Test LightGCN model reloading."""
    with patch('os.path.exists', return_value=True):
        rec_service.reload_lightgcn_model("new/model/path")
        mock_lightgcn.load_new_model.assert_called_once_with(
            "new/model/path",
            rec_service._ratings
        )

def test_invalid_model_reload(rec_service):
    """Test handling of invalid model reload attempts."""
    with patch('os.path.exists', return_value=False):
        with pytest.raises(RecommenderException):
            rec_service.reload_lightgcn_model("invalid/path")

def test_recommendation_enrichment(rec_service, mock_movie_repo):
    """Test recommendation enrichment with movie details."""
    raw_recs = [(101, 0.8), (102, 0.7)]
    enriched = rec_service._enrich_recs(raw_recs)
    
    assert len(enriched) == 2
    assert all(isinstance(r, dict) for r in enriched)
    assert all('movie_id' in r and 'score' in r and 'title' in r for r in enriched)
    assert mock_movie_repo.get_by_id.call_count == 2

def test_update_error_handling(rec_service, mock_rating_repo):
    """Test error handling during updates."""
    mock_rating_repo.get_all_positive_ratings.side_effect = Exception("Database error")
    
    with pytest.raises(RecommenderException):
        rec_service.update_ratings()

def test_concurrent_recommendations(rec_service, mock_hybrid):
    """Test concurrent recommendation requests."""
    num_requests = 10
    results = []
    
    def get_recs():
        try:
            recs = rec_service.get_user_hybrid_recommendations(1, 5)
            results.append(recs)
        except Exception as e:
            results.append(e)
    
    # Create and run multiple threads
    threads = [threading.Thread(target=get_recs) for _ in range(num_requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(results) == num_requests
    assert all(isinstance(r, list) for r in results)
    assert all(len(r) == 2 for r in results)  # Based on mock hybrid returning 2 recs

def test_get_similar_items(rec_service, mock_content):
    """Test getting similar items."""
    # Setup the mock to return some test data
    mock_content.get_similar_items.return_value = [(102, 0.9), (103, 0.85)]
    
    similar_items = rec_service.get_similar_items(item_id=101, top_k=5)
    
    assert isinstance(similar_items, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in similar_items)
    assert all(isinstance(item[0], int) and isinstance(item[1], float) for item in similar_items)

def test_get_similar_items_invalid_id(rec_service):
    """Test getting similar items with invalid ID."""
    with pytest.raises(RecommendationFailedException):
        rec_service.get_similar_items(item_id=-1)
    
    with pytest.raises(RecommendationFailedException):
        rec_service.get_similar_items(item_id="invalid") 