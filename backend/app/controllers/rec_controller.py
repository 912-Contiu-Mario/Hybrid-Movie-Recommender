import time
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List

from app.domain.models import User
from app.domain.dto import (
    RecommendationParams, 
    MovieRecommendation, 
    SimilarMovieParams,
    ModelReloadRequest
)

from app.service.dependencies import get_rec_service
from app.service.rec_service import RecService
from app.auth.dependencies import get_current_active_user
from app.exceptions.recommender import (
    ResourceNotFoundException,
    RecommendationFailedException,
    InvalidRequestException
)
import os
from app.config import paths

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
    responses={404: {"description": "Not found"}}
)

@router.get("/lightgcn/{user_id}", response_model=List[MovieRecommendation])
def get_lightgcn_recommendations(
    user_id: int, 
    num_recs: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    rec_service: RecService = Depends(get_rec_service)
):
    try:
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view recommendations for other users"
            )
            
        recs = rec_service.get_user_lightgcn_recommendations(user_id, num_recs)
        return [
            MovieRecommendation(
                id=rec["id"],
                tmdb_id=rec["tmdb_id"],
                title=rec["title"],
                score=rec["score"]
            ) 
            for rec in recs
        ]
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RecommendationFailedException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/content/{user_id}", response_model=List[MovieRecommendation])
def get_content_recommendations(
    user_id: int, 
    num_recs: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    rec_service: RecService = Depends(get_rec_service)
):
    try:
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view recommendations for other users"
            )
            
        recs = rec_service.get_user_content_recommendations(user_id, num_recs)
        return [
            MovieRecommendation(
                id=rec["id"],
                tmdb_id=rec["tmdb_id"],
                title=rec["title"],
                score=rec["score"]
            ) 
            for rec in recs
        ]
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RecommendationFailedException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hybrid/{user_id}", response_model=List[MovieRecommendation])
def get_hybrid_recommendations(
    user_id: int, 
    num_recs: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    rec_service: RecService = Depends(get_rec_service)
):
    try:
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view recommendations for other users"
            )
            
        recs = rec_service.get_user_hybrid_recommendations(user_id, num_recs)
        return [
            MovieRecommendation(
                id=rec["id"],
                tmdb_id=rec["tmdb_id"],
                title=rec["title"],
                score=rec["score"]
            ) 
            for rec in recs
        ]
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RecommendationFailedException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/similar-movies/{movie_id}", response_model=List[MovieRecommendation])
def get_similar_movies(
    movie_id: int,
    num_similar: int = Query(10, ge=1, le=100),
    rec_service: RecService = Depends(get_rec_service)
):
    try:
        similar_movies = rec_service.get_similar_items(movie_id, num_similar)
        return [
            MovieRecommendation(
                id=movie["id"],
                tmdb_id=movie["tmdb_id"],
                title=movie["title"],
                score=movie["score"]
            ) 
            for movie in similar_movies
        ]
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RecommendationFailedException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reload")
def reload(
    model_data: ModelReloadRequest,
    rec_service: RecService = Depends(get_rec_service),
    current_user: User = Depends(get_current_active_user)
):
    try:
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
            )
            
        model_dir = os.path.join(paths.LIGHTGCN_DIR, model_data.version)
        rec_service.reload_lightgcn_model(model_dir)
        return {"message": f"Model reloaded successfully from {model_dir}"}
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RecommendationFailedException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while reloading the model: {str(e)}"
        )
    

@router.post("/generate-recs-for-several-items", response_model=List[MovieRecommendation])
def generate_recs_for_several_items(
    item_ids: List[int],
    num_recs: int = Query(10, ge=1, le=100),
    rec_service: RecService = Depends(get_rec_service)
):
    try:
        similar_movies = rec_service.generate_similar_items(item_ids, num_recs)
        return [
            MovieRecommendation(
                id=movie["id"],
                tmdb_id=movie["tmdb_id"],
                title=movie["title"],
                score=movie["score"]
            )
            for movie in similar_movies
        ]
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
