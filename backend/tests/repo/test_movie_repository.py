import pytest
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, MovieORM
from app.domain.models import Movie
from app.repositories.implementation.sql_alchemy_movie_repo import SQLAlchemyMovieRepo
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException,
    InvalidEntityDataException
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
def movie_repo(session):
    """Create a movie repository instance."""
    return SQLAlchemyMovieRepo(session)


@pytest.fixture
def test_movies(session):
    """Create test movies in the database."""
    movies = [
        MovieORM(
            id=1,
            tmdb_id=101,
            title="The Matrix",
            genres=json.dumps([
                {"id": 878, "name": "Science Fiction"},
                {"id": 28, "name": "Action"}
            ])
        ),
        MovieORM(
            id=2,
            tmdb_id=102,
            title="Titanic",
            genres=json.dumps([
                {"id": 18, "name": "Drama"},
                {"id": 10749, "name": "Romance"}
            ])
        )
    ]
    session.add_all(movies)
    session.commit()
    return movies


def test_get_by_id_existing_movie(movie_repo, test_movies):
    """Test retrieving an existing movie by ID."""
    movie = movie_repo.get_by_id(1)
    assert movie is not None
    assert movie.id == 1
    assert movie.title == "The Matrix"
    assert movie.tmdb_id == 101
    assert movie.genres == ["Science Fiction", "Action"]


def test_get_by_id_non_existent_movie(movie_repo):
    """Test retrieving a non-existent movie by ID."""
    movie = movie_repo.get_by_id(999)
    assert movie is None


def test_get_all_items_with_movies(movie_repo, test_movies):
    """Test retrieving all movies when movies exist."""
    movies = movie_repo.get_all_items()
    assert len(movies) == 2
    
    # Verify each movie's properties
    matrix = next(m for m in movies if m.title == "The Matrix")
    assert matrix.genres == ["Science Fiction", "Action"]
    
    titanic = next(m for m in movies if m.title == "Titanic")
    assert titanic.genres == ["Drama", "Romance"]


def test_get_all_items_empty_db(movie_repo):
    """Test retrieving all movies from an empty database."""
    movies = movie_repo.get_all_items()
    assert len(movies) == 0
    assert isinstance(movies, list)


def test_get_by_id_invalid_input(movie_repo, session):
    """Test handling of invalid input in get_by_id."""
    with pytest.raises(RepositoryOperationException):
        movie_repo.get_by_id("invalid_id")  # Passing string instead of int


def test_domain_model_conversion(movie_repo, test_movies):
    """Test that movies are correctly converted to domain models."""
    movie = movie_repo.get_by_id(1)
    assert movie.__class__.__name__ == "Movie"  # Verify it's a domain model
    assert isinstance(movie.id, int)
    assert isinstance(movie.title, str)
    assert isinstance(movie.tmdb_id, int)
    assert isinstance(movie.genres, list)  # genres should be a list now
    assert all(isinstance(genre, str) for genre in movie.genres)  # each genre should be a string


def test_get_by_id_genre_conversion(movie_repo, test_movies):
    """Test that genres are correctly converted from ORM to domain model."""
    movie = movie_repo.get_by_id(1)
    assert movie is not None
    assert movie.id == 1
    assert movie.title == "The Matrix"
    assert movie.tmdb_id == 101
    assert isinstance(movie.genres, list)
    assert movie.genres == ["Science Fiction", "Action"]


def test_convert_domain_to_orm(movie_repo):
    """Test converting a domain model to ORM with proper genre format."""
    # Create a domain model
    movie = Movie(
        title="Inception",
        tmdb_id=103,
        genres=["Science Fiction", "Action", "Thriller"],
        id=3
    )
    
    # Convert to ORM
    movie_orm = movie_repo._to_orm(movie)
    
    # Verify the conversion
    assert movie_orm.title == "Inception"
    assert movie_orm.tmdb_id == 103
    
    # Check that genres are properly converted to JSON
    genres = json.loads(movie_orm.genres)
    assert isinstance(genres, list)
    assert len(genres) == 3
    assert all("name" in genre for genre in genres)
    assert [genre["name"] for genre in genres] == ["Science Fiction", "Action", "Thriller"]


def test_invalid_genre_json(session, movie_repo):
    """Test handling of invalid JSON in genres field."""
    # Create a movie with invalid JSON in genres
    invalid_movie = MovieORM(
        id=4,
        tmdb_id=104,
        title="Invalid Movie",
        genres="invalid json"
    )
    session.add(invalid_movie)
    session.commit()

    # Attempt to convert to domain model should raise an exception
    with pytest.raises(InvalidEntityDataException) as exc_info:
        movie_repo.get_by_id(4)
    assert "Failed to convert movie data" in str(exc_info.value)


def test_missing_genre_name(session, movie_repo):
    """Test handling of missing 'name' field in genre data."""
    # Create a movie with genres missing the 'name' field
    invalid_movie = MovieORM(
        id=5,
        tmdb_id=105,
        title="Missing Name Movie",
        genres=json.dumps([{"id": 28}, {"id": 12}])  # Missing 'name' field
    )
    session.add(invalid_movie)
    session.commit()

    # Attempt to convert to domain model should raise an exception
    with pytest.raises(InvalidEntityDataException) as exc_info:
        movie_repo.get_by_id(5)
    assert "Failed to convert movie data" in str(exc_info.value)


def test_get_all_items_genre_conversion(movie_repo, test_movies):
    """Test that genres are correctly converted for all movies."""
    movies = movie_repo.get_all_items()
    assert len(movies) == 2
    
    # Check first movie
    matrix = next(m for m in movies if m.title == "The Matrix")
    assert matrix.genres == ["Science Fiction", "Action"]
    
    # Check second movie
    titanic = next(m for m in movies if m.title == "Titanic")
    assert titanic.genres == ["Drama", "Romance"] 