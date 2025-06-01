from sqlalchemy.orm import Session
from typing import List, Optional
import json

from app.db.models import MovieORM
from app.domain.models import Movie
from app.repositories.interface.movie_repository import MovieRepository
from app.exceptions.repository import (
    RepositoryOperationException,
    InvalidEntityDataException
)


class SQLAlchemyMovieRepo(MovieRepository):
    def __init__(self, session: Session):
        self.session = session

    def _to_domain(self, movie_orm: MovieORM) -> Movie:
        try:
            try:
                # Try standard JSON parsing first
                genres_dict = json.loads(movie_orm.genres)
            except json.JSONDecodeError:
                # If that fails, try to handle Python literal style with single quotes
                import ast
                genres_dict = ast.literal_eval(movie_orm.genres)
            
            genre_names = [genre['name'] for genre in genres_dict]
            
            return Movie(
                id=movie_orm.id,
                title=movie_orm.title,
                tmdb_id=movie_orm.tmdb_id,
                genres=genre_names
            )
        except Exception as e:
            raise InvalidEntityDataException("Movie", f"Failed to convert movie data: {str(e)}")

    def _to_orm(self, movie: Movie) -> MovieORM:
        try:
            # Convert the list of genre names to the required format
            genres_dict = [{"name": genre} for genre in movie.genres]
            genres_json = json.dumps(genres_dict)
            
            return MovieORM(
                id=movie.id,
                title=movie.title,
                tmdb_id=movie.tmdb_id,
                genres=genres_json
            )
        except Exception as e:
            raise InvalidEntityDataException("Movie", f"Failed to convert to ORM: {str(e)}")

    def get_by_id(self, movie_id: int) -> Optional[Movie]:
        try:
            if not isinstance(movie_id, int):
                raise RepositoryOperationException(f"Invalid movie_id type. Expected int, got {type(movie_id)}")
            
            movie_orm = self.session.query(MovieORM).get(movie_id)
            if not movie_orm:
                return None
            return self._to_domain(movie_orm)
        except InvalidEntityDataException:
            raise
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get movie by ID: {str(e)}")
    

    def get_all_items(self) -> List[Movie]:
        try:
            movies_orm = self.session.query(MovieORM).all()
            return [self._to_domain(movie_orm) for movie_orm in movies_orm]
        except InvalidEntityDataException:
            raise
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get all movies: {str(e)}")
        
    def get_by_TMDB_ids(self, tmdb_ids: List[int]) -> List[Movie]:
        try:
            movies_orm = self.session.query(MovieORM).filter(MovieORM.tmdb_id.in_(tmdb_ids)).all()
            return [self._to_domain(movie_orm) for movie_orm in movies_orm]
        except InvalidEntityDataException:
            raise
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get movies by TMDB IDs: {str(e)}")
