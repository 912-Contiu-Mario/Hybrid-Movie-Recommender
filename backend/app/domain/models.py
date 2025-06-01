from datetime import datetime
from typing import Optional, List

class Rating:
    def __init__(
        self,
        user_id: int,
        movie_id: int,
        rating: float,
        timestamp: datetime = None
    ):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
        self.timestamp = timestamp

class User:
    def __init__(
        self,
        username: str,
        email: str,
        hashed_password: str,
        id: Optional[int] = None,
        is_active: bool = True,
        is_admin: bool = False,
        is_test: bool = False,
        created_at: datetime = None
    ):
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.id = id
        self.is_active = is_active
        self.is_admin = is_admin
        self.is_test = is_test
        self.created_at = created_at

class Movie:
    def __init__(
        self,
        title: str,
        genres: List[str],
        tmdb_id: int,
        id: Optional[int] = None
    ):
        self.title = title
        self.genres = genres
        self.tmdb_id = tmdb_id
        self.id = id

