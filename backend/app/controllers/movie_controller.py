from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.domain.dto import MovieIdResponse, MovieResponse, TMDBIdsRequest
from app.domain.models import Movie
from app.service.dependencies import get_movie_service
from app.service.movie_service import MovieService





router = APIRouter(
    prefix="/movies",
    tags=["Movies"],
    responses={404: {"description": "Not found"}}
)


@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie(
    movie_id: int,
    movie_service: MovieService = Depends(get_movie_service)
):
    
    try:
        movie: Movie = movie_service.get_movie(movie_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


    movie_response = MovieResponse(
        id=movie.id,
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        genres=movie.genres
    )

    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    

    return movie_response


@router.post("/filter-existing-TMDB-ids", response_model=List[MovieIdResponse])
def filter_existing_movies(
    request: TMDBIdsRequest,
    movie_service: MovieService = Depends(get_movie_service)
):
    try:
        existing_movie_ids: List[dict] = movie_service.filter_existing_movies_by_TMDB_ids(request.tmdb_ids)
        return existing_movie_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
