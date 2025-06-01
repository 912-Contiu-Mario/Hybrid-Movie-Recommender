from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from app.domain.models import Rating

class RatingRepository(ABC):
    @abstractmethod
    def get_user_ratings(self, user_id: int) -> Dict[int, float]:
        pass

    @abstractmethod
    def add_rating(self, rating: Rating) -> None:
        pass

    @abstractmethod
    def get_user_positive_ratings(self, user_id: int, rating_threshold: float) -> List["Rating"]:
        pass

    @abstractmethod
    def get_all_positive_ratings(self, rating_threshold: float) -> List["Rating"]:
        pass

    @abstractmethod
    def get_all_ratings(self) -> List["Rating"]:
        pass 

    @abstractmethod
    def get_by_user_id_and_movie_id(self, user_id: int, movie_id: int) -> Optional["Rating"]:
        pass 

    @abstractmethod
    def delete_by_user_id_and_movie_id(self, user_id: int, movie_id: int) -> bool:
        pass

    @abstractmethod
    def update_rating(self, rating: Rating) -> None:
        pass
