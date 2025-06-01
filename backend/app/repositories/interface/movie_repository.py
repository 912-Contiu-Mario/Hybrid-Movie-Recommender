from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models import Movie


class MovieRepository(ABC):
    @abstractmethod
    def get_by_id(self, movie_id: int) -> Optional["Movie"]:
        pass

    @abstractmethod
    def get_all_items(self) -> List["Movie"]:
        pass 

    @abstractmethod
    def get_by_TMDB_ids(self, tmdb_ids: List[int]) -> List["Movie"]:
        pass