from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any

from app.domain.models import User
from app.domain.dto import UserProfile, RatingCreate, RatingResponse
from app.auth.dependencies import get_current_active_user
from app.service.dependencies import get_rating_service
from app.service.rating_service import RatingService

# TODO: add user DTO
# TODO: more custom exception handling.


router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}}
)

@router.get("/me", response_model=UserProfile)
def read_user_me(current_user: User = Depends(get_current_active_user)):
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_admin=current_user.is_admin
    )

@router.post("/ratings", status_code=status.HTTP_201_CREATED)
def rate_movie(
    rating_data: RatingCreate,
    current_user: User = Depends(get_current_active_user),
    rating_service: RatingService = Depends(get_rating_service)
):
    try:
        rating_service.add_rating(current_user.id, rating_data.movie_id, rating_data.rating)
        
        return {"status": "success", "message": "Rating saved successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save rating: {str(e)}"
        )

@router.get("/ratings", response_model=List[RatingResponse])
def get_user_ratings(
    current_user: User = Depends(get_current_active_user),
    rating_service: RatingService = Depends(get_rating_service)
):
    ratings = rating_service.get_user_ratings(current_user.id)
    return [
        RatingResponse(
            movie_id=rating.movie_id,
            rating=rating.rating,
            timestamp=rating.timestamp
        )
        for rating in ratings
    ]

@router.delete("/ratings/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(
    movie_id: int,
    current_user: User = Depends(get_current_active_user),
    rating_service: RatingService = Depends(get_rating_service)
):
    
    success = rating_service.remove_rating(current_user.id, movie_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found"
        ) 