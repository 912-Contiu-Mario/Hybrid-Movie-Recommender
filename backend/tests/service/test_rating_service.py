import pytest
from datetime import datetime
from unittest.mock import Mock

from app.domain.models import User, Movie, Rating
from app.service.rating_service import RatingService
from app.exceptions.rating import RatingServiceException, InvalidRequestException
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException
)


@pytest.fixture
def mock_repos():
    """Create mock repositories."""
    rating_repo = Mock()
    user_repo = Mock()
    movie_repo = Mock()
    return rating_repo, user_repo, movie_repo


@pytest.fixture
def rating_service(mock_repos):
    """Create a rating service with mock repositories."""
    rating_repo, user_repo, movie_repo = mock_repos
    return RatingService(rating_repo, user_repo, movie_repo)


@pytest.fixture
def test_data():
    """Create test data."""
    now = datetime.now()
    
    user = User(
        id=1,
        username="alice",
        email="alice@test.com",
        hashed_password="hashed_pw",
        is_active=True,
        created_at=now
    )
    
    movie = Movie(
        id=1,
        title="The Matrix",
        tmdb_id=101,
        genres=["Action", "Sci-Fi"]
    )
    
    rating = Rating(
        user_id=1,
        movie_id=1,
        rating=4.5,
        timestamp=now
    )
    
    return {
        "user": user,
        "movie": movie,
        "rating": rating,
        "timestamp": now
    }


def test_add_rating_success(rating_service, mock_repos, test_data):
    """Test successfully adding a new rating."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    movie = test_data["movie"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    movie_repo.get_by_id.return_value = movie
    rating_repo.get_by_user_id_and_movie_id.return_value = None
    rating_repo.add_rating.return_value = test_data["rating"]
    
    # Test
    result = rating_service.add_rating(user.id, movie.id, 4.5)
    
    # Verify
    assert result.user_id == user.id
    assert result.movie_id == movie.id
    assert result.rating == 4.5
    user_repo.get_by_id.assert_called_once_with(user.id)
    movie_repo.get_by_id.assert_called_once_with(movie.id)


def test_add_rating_user_not_found(rating_service, mock_repos, test_data):
    """Test adding a rating with non-existent user."""
    rating_repo, user_repo, movie_repo = mock_repos
    
    # Setup mocks
    user_repo.get_by_id.return_value = None
    
    # Test
    with pytest.raises(InvalidRequestException, match="User with ID 1 not found"):
        rating_service.add_rating(1, 1, 4.5)


def test_add_rating_movie_not_found(rating_service, mock_repos, test_data):
    """Test adding a rating with non-existent movie."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    movie_repo.get_by_id.return_value = None
    
    # Test
    with pytest.raises(InvalidRequestException, match="Movie with ID 1 not found"):
        rating_service.add_rating(1, 1, 4.5)


def test_add_rating_update_existing(rating_service, mock_repos, test_data):
    """Test updating an existing rating."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    movie = test_data["movie"]
    existing_rating = test_data["rating"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    movie_repo.get_by_id.return_value = movie
    rating_repo.get_by_user_id_and_movie_id.return_value = existing_rating
    
    updated_rating = Rating(user_id=1, movie_id=1, rating=3.5, timestamp=datetime.now())
    rating_repo.update_rating.return_value = updated_rating
    
    # Test
    result = rating_service.add_rating(user.id, movie.id, 3.5)
    
    # Verify
    assert result.rating == 3.5
    rating_repo.update_rating.assert_called_once()


def test_get_user_ratings_success(rating_service, mock_repos, test_data):
    """Test successfully getting user ratings."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    ratings = [test_data["rating"]]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    rating_repo.get_user_ratings.return_value = ratings
    
    # Test
    result = rating_service.get_user_ratings(user.id)
    
    # Verify
    assert len(result) == 1
    assert result[0].user_id == user.id
    assert result[0].rating == 4.5


def test_get_user_ratings_user_not_found(rating_service, mock_repos):
    """Test getting ratings for non-existent user."""
    rating_repo, user_repo, movie_repo = mock_repos
    
    # Setup mocks
    user_repo.get_by_id.return_value = None
    
    # Test
    with pytest.raises(InvalidRequestException, match="User with ID 1 not found"):
        rating_service.get_user_ratings(1)


def test_remove_rating_success(rating_service, mock_repos, test_data):
    """Test successfully removing a rating."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    rating = test_data["rating"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    rating_repo.get_by_user_id_and_movie_id.return_value = rating
    rating_repo.delete_by_user_id_and_movie_id.return_value = True
    
    # Test
    result = rating_service.remove_rating(user.id, rating.movie_id)
    
    # Verify
    assert result is True
    rating_repo.delete_by_user_id_and_movie_id.assert_called_once_with(user.id, rating.movie_id)


def test_remove_rating_user_not_found(rating_service, mock_repos):
    """Test removing a rating for non-existent user."""
    rating_repo, user_repo, movie_repo = mock_repos
    
    # Setup mocks
    user_repo.get_by_id.return_value = None
    
    # Test
    with pytest.raises(InvalidRequestException, match="User with ID 1 not found"):
        rating_service.remove_rating(1, 1)


def test_remove_rating_not_found(rating_service, mock_repos, test_data):
    """Test removing a non-existent rating."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    rating_repo.get_by_user_id_and_movie_id.return_value = None
    
    # Test
    with pytest.raises(InvalidRequestException, match="Rating for user 1 and movie 1 not found"):
        rating_service.remove_rating(1, 1)


def test_remove_rating_delete_failed(rating_service, mock_repos, test_data):
    """Test when rating deletion fails."""
    rating_repo, user_repo, movie_repo = mock_repos
    user = test_data["user"]
    rating = test_data["rating"]
    
    # Setup mocks
    user_repo.get_by_id.return_value = user
    rating_repo.get_by_user_id_and_movie_id.return_value = rating
    rating_repo.delete_by_user_id_and_movie_id.return_value = False
    
    # Test
    with pytest.raises(RatingServiceException, match="Failed to delete rating"):
        rating_service.remove_rating(1, 1) 