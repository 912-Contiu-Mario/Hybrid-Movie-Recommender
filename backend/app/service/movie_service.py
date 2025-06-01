from typing import List
from app.domain.models import Movie
from app.repositories.interface.movie_repository import MovieRepository


class MovieService:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository


    def filter_existing_movies_by_TMDB_ids(self, tmdb_ids: List[int]) -> List[dict]:

        existing_movies: List[Movie] = self.movie_repository.get_by_TMDB_ids(tmdb_ids)
        # Return both internal movie IDs and TMDB IDs
        return [{"id": movie.id, "tmdb_id": movie.tmdb_id} for movie in existing_movies]
    

    def get_movie(self, movie_id: int) -> Movie:
        movie: Movie = self.movie_repository.get_by_id(movie_id)
        if movie is None:
            raise Exception("Movie not found")
        return movie
