from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.repositories import SQLAlchemyUserRepo, SQLAlchemyRatingRepo, SQLAlchemyMovieRepo
from app.service.auth_service import AuthService
from app.service.movie_service import MovieService
from app.service.rating_service import RatingService
from app.service.rec_service import RecService

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    return AuthService(SQLAlchemyUserRepo(db))

def get_rating_service(db: Session = Depends(get_db)) -> RatingService:
    return RatingService(
        rating_repo=SQLAlchemyRatingRepo(db),
        user_repo=SQLAlchemyUserRepo(db),
        movie_repo=SQLAlchemyMovieRepo(db)
    )

def get_rec_service() -> RecService:
    from app.main import app
    return app.state.rec_service 

def get_movie_service(db: Session = Depends(get_db)) -> MovieService:
    return MovieService(SQLAlchemyMovieRepo(db))