from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
import time
from sqlalchemy.exc import IntegrityError

from app.db.models import RatingORM
from app.domain.models import Rating
from app.repositories.interface.rating_repository import RatingRepository
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException,
    InvalidEntityDataException
)


class SQLAlchemyRatingRepo(RatingRepository):
    def __init__(self, session: Session):
        self.session = session

    def _to_domain(self, rating_orm: RatingORM) -> Rating:
        try:
            return Rating(
                user_id=rating_orm.user_id,
                movie_id=rating_orm.movie_id,
                rating=rating_orm.rating,
                timestamp=rating_orm.timestamp
            )
        except Exception as e:
            raise InvalidEntityDataException("Rating", f"Failed to convert rating data: {str(e)}")
    
    
    def _to_orm(self, rating: Rating) -> RatingORM:
        try:
            return RatingORM(
                user_id=rating.user_id,
                movie_id=rating.movie_id,
                rating=rating.rating,
                timestamp=rating.timestamp
            )
        except Exception as e:
            raise InvalidEntityDataException("Rating", f"Failed to convert to ORM: {str(e)}")

    def get_user_ratings(self, user_id: int) -> List[Rating]:
        try:
            ratings_orm = self.session.query(RatingORM).filter(
                RatingORM.user_id == user_id
            ).all()
            return [self._to_domain(r) for r in ratings_orm]
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get user ratings: {str(e)}")


    def get_user_positive_ratings(self, user_id: int, rating_threshold: float) -> List[Rating]:
        try:
            ratings_orm = self.session.query(RatingORM).filter(
                (RatingORM.user_id == user_id) & 
                (RatingORM.rating >= rating_threshold)
            ).all()
            return [self._to_domain(r) for r in ratings_orm]
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get user positive ratings: {str(e)}")

    def get_all_positive_ratings(self, rating_threshold: float) -> List[Rating]:
        try:
            ratings_orm = self.session.query(RatingORM).filter(
                RatingORM.rating >= rating_threshold
            ).all()
            return [self._to_domain(r) for r in ratings_orm]
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get all positive ratings: {str(e)}")

    def get_all_ratings(self) -> List[Rating]:
        try:
            ratings_orm = self.session.query(RatingORM).all()
            return [self._to_domain(r) for r in ratings_orm]
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get all ratings: {str(e)}")

    def add_rating(self, rating: Rating) -> Rating:
        try:
            existing_rating = self.get_by_user_id_and_movie_id(rating.user_id, rating.movie_id)
            if existing_rating:
                raise DuplicateEntityException("Rating", "user_id and movie_id", f"{rating.user_id}, {rating.movie_id}")
            
            new_rating = self._to_orm(rating)
            self.session.add(new_rating)
            self.session.commit()
            return self._to_domain(new_rating)
        except DuplicateEntityException:
            raise
        except Exception as e:
            raise RepositoryOperationException(f"Failed to add rating: {str(e)}")

    def get_by_user_id_and_movie_id(self, user_id: int, movie_id: int) -> Optional[Rating]:
        try:
            rating_orm = self.session.query(RatingORM).filter(
                RatingORM.user_id == user_id,
                RatingORM.movie_id == movie_id
            ).first()
            return self._to_domain(rating_orm) if rating_orm else None
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get rating by user and movie: {str(e)}")
    
    def delete_by_user_id_and_movie_id(self, user_id: int, movie_id: int) -> bool:
        try:
            rating_orm = self.session.query(RatingORM).filter(
                RatingORM.user_id == user_id,
                RatingORM.movie_id == movie_id
            ).first()
            if rating_orm:
                self.session.delete(rating_orm)
                self.session.commit()
                return True
            return False
        except Exception as e:
            raise RepositoryOperationException(f"Failed to delete rating: {str(e)}")

    def update_rating(self, rating: Rating) -> Rating:
        try:
            rating_orm = self.session.query(RatingORM).filter(
                RatingORM.user_id == rating.user_id,
                RatingORM.movie_id == rating.movie_id
            ).first()
            if not rating_orm:
                raise EntityNotFoundException(f"Rating for user {rating.user_id} and movie {rating.movie_id}")
            
            rating_orm.rating = rating.rating
            rating_orm.timestamp = datetime.now()
            self.session.commit()
            return self._to_domain(rating_orm)
        except EntityNotFoundException:
            raise
        except Exception as e:
            raise RepositoryOperationException(f"Failed to update rating: {str(e)}")