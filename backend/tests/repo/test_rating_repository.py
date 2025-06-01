import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, UserORM, MovieORM, RatingORM
from app.domain.models import Rating
from app.repositories.implementation.sql_alchemy_rating_repo import SQLAlchemyRatingRepo
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException
)


@pytest.fixture
def session():
    """Create a fresh in-memory SQLite database for each test."""
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture
def rating_repo(session):
    """Create a rating repository instance."""
    return SQLAlchemyRatingRepo(session)


@pytest.fixture
def test_data(session):
    """Create test users, movies and ratings."""
    # Create users
    user1 = UserORM(id=1, username="alice", hashed_password="hashed_pw1")
    user2 = UserORM(id=2, username="bob", hashed_password="hashed_pw2")
    
    # Create movies
    movie1 = MovieORM(id=1, tmdb_id=101, title="The Matrix", genres="Action|Sci-Fi")
    movie2 = MovieORM(id=2, tmdb_id=102, title="Titanic", genres="Drama|Romance")
    movie3 = MovieORM(id=3, tmdb_id=103, title="The Dark Knight", genres="Action|Crime|Drama")
    
    session.add_all([user1, user2, movie1, movie2, movie3])
    session.commit()

    # Add ratings
    now = datetime.now()
    ratings = [
        RatingORM(user_id=1, movie_id=1, rating=4.5, timestamp=now),  # Alice rates Matrix
        RatingORM(user_id=1, movie_id=2, rating=2.0, timestamp=now),  # Alice rates Titanic
        RatingORM(user_id=2, movie_id=1, rating=3.5, timestamp=now),  # Bob rates Matrix
        RatingORM(user_id=2, movie_id=3, rating=5.0, timestamp=now),  # Bob rates Dark Knight
    ]
    session.add_all(ratings)
    session.commit()

    return {
        "users": [user1, user2],
        "movies": [movie1, movie2, movie3],
        "ratings": ratings,
        "timestamp": now
    }


def test_get_user_ratings(rating_repo, test_data):
    """Test getting all ratings for a user."""
    ratings = rating_repo.get_user_ratings(1)  # Alice's ratings
    assert len(ratings) == 2
    assert all(isinstance(r, Rating) for r in ratings)
    assert any(r.rating == 4.5 for r in ratings)
    assert any(r.rating == 2.0 for r in ratings)


def test_get_user_positive_ratings(rating_repo, test_data):
    """Test getting positive ratings (>= threshold) for a user."""
    ratings = rating_repo.get_user_positive_ratings(1, rating_threshold=3.0)
    assert len(ratings) == 1
    assert ratings[0].rating == 4.5


def test_get_all_positive_ratings(rating_repo, test_data):
    """Test getting all positive ratings across users."""
    ratings = rating_repo.get_all_positive_ratings(rating_threshold=3.0)
    assert len(ratings) == 3  # 4.5, 3.5, and 5.0
    assert all(r.rating >= 3.0 for r in ratings)


def test_get_all_ratings(rating_repo, test_data):
    """Test getting all ratings."""
    ratings = rating_repo.get_all_ratings()
    assert len(ratings) == 4
    assert all(isinstance(r, Rating) for r in ratings)


def test_add_rating(rating_repo, test_data):
    """Test adding a new rating."""
    new_rating = Rating(user_id=1, movie_id=3, rating=4.0, timestamp=datetime.now())
    saved_rating = rating_repo.add_rating(new_rating)
    
    assert saved_rating.user_id == new_rating.user_id
    assert saved_rating.movie_id == new_rating.movie_id
    assert saved_rating.rating == new_rating.rating


def test_add_duplicate_rating(rating_repo, test_data):
    """Test adding a duplicate rating raises exception."""
    duplicate_rating = Rating(user_id=1, movie_id=1, rating=3.0, timestamp=datetime.now())
    with pytest.raises(DuplicateEntityException):
        rating_repo.add_rating(duplicate_rating)


def test_get_by_user_id_and_movie_id(rating_repo, test_data):
    """Test getting a specific rating by user and movie ID."""
    rating = rating_repo.get_by_user_id_and_movie_id(1, 1)
    assert rating is not None
    assert rating.rating == 4.5


def test_get_by_user_id_and_movie_id_nonexistent(rating_repo, test_data):
    """Test getting a nonexistent rating returns None."""
    rating = rating_repo.get_by_user_id_and_movie_id(1, 999)
    assert rating is None


def test_delete_rating(rating_repo, test_data):
    """Test deleting a rating."""
    success = rating_repo.delete_by_user_id_and_movie_id(1, 1)
    assert success is True
    
    # Verify it's gone
    rating = rating_repo.get_by_user_id_and_movie_id(1, 1)
    assert rating is None


def test_delete_nonexistent_rating(rating_repo, test_data):
    """Test deleting a nonexistent rating returns False."""
    success = rating_repo.delete_by_user_id_and_movie_id(999, 999)
    assert success is False


def test_update_rating(rating_repo, test_data):
    """Test updating an existing rating."""
    updated_rating = Rating(user_id=1, movie_id=1, rating=3.0, timestamp=datetime.now())
    result = rating_repo.update_rating(updated_rating)
    
    assert result.rating == 3.0
    assert result.user_id == 1
    assert result.movie_id == 1


def test_update_nonexistent_rating(rating_repo, test_data):
    """Test updating a nonexistent rating raises exception."""
    nonexistent_rating = Rating(user_id=999, movie_id=999, rating=3.0, timestamp=datetime.now())
    with pytest.raises(EntityNotFoundException):
        rating_repo.update_rating(nonexistent_rating) 