import os
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging

from app.db.database import engine, Base, get_db
from app.scripts.import_movies import import_all
from app.service.rec_service import RecService
import asyncio
from datetime import datetime, timedelta
from app.config import get_recommender_config, MOVIES_PATH, RATINGS_SMALL_PATH, LIGHTGCN_MODEL_DATA_DIR, COMBINED_EMBEDDINGS_PATH, paths
from app.config.logging import setup_logging
from app.controllers.rec_controller import router as rec_router
from app.controllers.auth_controller import router as auth_router
from app.controllers.user_controller import router as user_router
from app.controllers.movie_controller import router as movie_router

from app.recommenders.content_rec import ContentRecommender
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.recommenders.hybrid_rec import HybridRecommender
from app.exceptions.recommender import (
    InvalidRequestException,
    RecommendationFailedException
)
from app.repositories import SQLAlchemyRatingRepo, SQLAlchemyMovieRepo, SQLAlchemyUserRepo

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommendation API",
    description="API for hybrid movie recommendations",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

# include controllers
app.include_router(rec_router)
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(movie_router)

rec_service = None

def init_db():
    Base.metadata.create_all(bind=engine)

init_db()

# initialize the recommenders and service at startup
@app.on_event("startup")
async def startup_event():
    global rec_service
    
    # get a db session for initialization
    db = next(get_db())
    try:
        # init repositories
        rating_repo = SQLAlchemyRatingRepo(db)
        user_repo = SQLAlchemyUserRepo(db)
        movie_repo = SQLAlchemyMovieRepo(db)

        # get recommender config
        recommender_config = get_recommender_config()

        # init rec service
        rec_service = RecService(
            rating_repository=rating_repo,
            user_repository=user_repo,
            movie_repository=movie_repo,
            config=recommender_config
        )

        # store in app state
        app.state.rec_service = rec_service

    except Exception as e:
        logger.error(f"Error initializing recommenders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initializing recommenders: {str(e)}"
        )
    finally:
        db.close()

@app.on_event("shutdown")
async def shutdown_event():
    # Stop the periodic update task
    if rec_service:
        await rec_service.stop_periodic_updates()
        logger.info("Stopped periodic update task")

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Debug endpoint for testing concurrency
@app.get("/debug/force-update")
async def force_update(background_tasks: BackgroundTasks):
    """Force a ratings update for testing purposes"""
    if not rec_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not initialized"
        )
    
    # Use the existing method from RecService for consistency
    background_tasks.add_task(rec_service._perform_update)
    return {"status": "success", "message": "Update started in background"}

@app.post("/import-data")
def import_data(
        purge: bool = False,
        db: Session = Depends(get_db)):
    try:
        import_all(purge=purge)

        return {"message": "Data import completed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error importing data: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"}