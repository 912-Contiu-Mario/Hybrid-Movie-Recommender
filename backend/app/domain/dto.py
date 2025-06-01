from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List
import re

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @field_validator('password')
    @classmethod
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr 


class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool

class RatingCreate(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=1, le=5.0)
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        return round(v * 2) / 2

class RatingResponse(BaseModel):
    movie_id: int
    rating: float
    timestamp: datetime


class RecommendationParams(BaseModel):
    user_id: int
    num_recs: int = Field(10, ge=1, le=100, description="Number of recommendations to return")

class MovieRecommendation(BaseModel):
    id: int
    tmdb_id: int
    title: str
    score: float

class SimilarMovieParams(BaseModel):
    movie_id: int
    num_similar: int = Field(10, ge=1, le=100, description="Number of similar movies to return")

class ModelReloadRequest(BaseModel):
    version: str = Field(..., description="Model version to load")


# Request model for the filter endpoint
class TMDBIdsRequest(BaseModel):
    tmdb_ids: List[int]


# Response model for the filter endpoint
class MovieIdResponse(BaseModel):
    id: int
    tmdb_id: int

class MovieResponse(BaseModel):
    id: int
    tmdb_id: int
    title: str
    genres: List[str]
