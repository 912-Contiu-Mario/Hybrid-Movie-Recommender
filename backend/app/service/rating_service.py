from typing import List
from datetime import datetime

from app.domain.models import Rating
from app.repositories import RatingRepository, UserRepository, MovieRepository
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException,
    InvalidEntityDataException
)
from app.exceptions.rating import (
    RatingServiceException,
    InvalidRequestException
)

class RatingService:
    def __init__(self, rating_repo: RatingRepository, user_repo: UserRepository, movie_repo: MovieRepository):
        self.rating_repo = rating_repo
        self.user_repo = user_repo
        self.movie_repo = movie_repo

    # def get_rating_by_id(self, rating_id: int) -> Rating:
    #     try:
    #         return self.rating_repo.get_by_id(rating_id)
    #     except RepositoryOperationException as e:
    #         raise
    #     except Exception as e:
    #         raise RatingServiceException(f"Unexpected error while getting rating by ID: {str(e)}")
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> Rating:
        try:
            # Check if user exists
            user = self.user_repo.get_by_id(user_id)
            if not user:
                raise InvalidRequestException(f"User with ID {user_id} not found")
            
            # Check if movie exists
            movie = self.movie_repo.get_by_id(movie_id)
            if not movie:
                raise InvalidRequestException(f"Movie with ID {movie_id} not found")
            
            rating_obj = Rating(user_id=user_id, movie_id=movie_id, rating=rating, timestamp=datetime.now())
            
            # Check if rating already exists
            existing_rating = self.rating_repo.get_by_user_id_and_movie_id(user_id, movie_id)
            if existing_rating:
                return self.rating_repo.update_rating(rating_obj)
            
            return self.rating_repo.add_rating(rating_obj)
        except InvalidRequestException as e:
            raise
        except RepositoryOperationException as e:
            raise
        except Exception as e:
            raise RatingServiceException(f"Unexpected error while adding rating: {str(e)}")

    def get_user_ratings(self, user_id: int) -> List[Rating]:
        try:
            # Check if user exists
            user = self.user_repo.get_by_id(user_id)
            if not user:
                raise InvalidRequestException(f"User with ID {user_id} not found")
            
            return self.rating_repo.get_user_ratings(user_id)
        except InvalidRequestException as e:
            raise
        except RepositoryOperationException as e:
            raise
        except Exception as e:
            raise RatingServiceException(f"Unexpected error while getting user ratings: {str(e)}")
    
    def remove_rating(self, user_id: int, movie_id: int) -> bool:
        try:
            # Check if user exists
            user = self.user_repo.get_by_id(user_id)
            if not user:
                raise InvalidRequestException(f"User with ID {user_id} not found")
            
            # Check if rating exists
            existing_rating = self.rating_repo.get_by_user_id_and_movie_id(user_id, movie_id)
            if not existing_rating:
                raise InvalidRequestException(f"Rating for user {user_id} and movie {movie_id} not found")
            
            success = self.rating_repo.delete_by_user_id_and_movie_id(user_id, movie_id)
            if not success:
                raise RatingServiceException("Failed to delete rating")
            
            return True
        except InvalidRequestException as e:
            raise
        except RepositoryOperationException as e:
            raise
        except Exception as e:
            raise RatingServiceException(f"Unexpected error while removing rating: {str(e)}")

