import time
from typing import List, Dict, Any, Tuple
import logging
import os
import asyncio

from app.domain.models import Rating

from app.recommenders.content_rec import ContentRecommender
from app.recommenders.hybrid_rec import HybridRecommender
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.repositories import UserRepository, MovieRepository, RatingRepository, SQLAlchemyRatingRepo
from app.exceptions.recommender import (
    RecommenderException,
    ResourceNotFoundException,
    InvalidRequestException,
    RecommendationFailedException
)
from app.db.database import SessionLocal

logger = logging.getLogger(__name__)

class RecService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecService, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 rating_repository: RatingRepository,
                 user_repository: UserRepository,
                 movie_repository: MovieRepository,
                 config: Dict[str, Any]
                 ):
        if not hasattr(self, 'initialized'):            
            self._content_rec: ContentRecommender = None
            self._hybrid_rec: HybridRecommender = None
            self._lightgcn_rec: LightGCNRecommender = None

            self._rating_repo = rating_repository
            self._movie_repo = movie_repository
            self._user_repo = user_repository

            self._ratings = []
            self._config = config
            self._rating_threshold = config.get('rating_threshold', 3.5)

            self._update_task = None
            self._update_interval = config.get('update_interval_seconds', 300)

            self._init_rec_services()
            
            self.initialized = True
            logger.info("RecService initialized successfully")


    def _init_rec_services(self):
        self._ratings = self._rating_repo.get_all_positive_ratings(self._rating_threshold)

        logger.info(f"Retrieved {len(self._ratings)} positive ratings")
        self._init_lightgcn_rec()
        self._init_content_rec()
        self._init_hybrid_rec()

        self.start_periodic_updates()

    def _init_lightgcn_rec(self):
        self._lightgcn_rec = LightGCNRecommender.get_instance(
            model_dir=self._config['lightgcn']["lightgcn_model_path"],
            ratings=self._ratings,
            device='cpu'
        )
        logger.info("LightGCN recommender initialized successfully")

    def _init_content_rec(self):
        self._content_rec = ContentRecommender.get_instance(
            embeddings_path=self._config['content']['embeddings_path'],
            min_rating=self._config['rating_threshold'],
            min_weight=self._config['content']['min_weight'],
            max_rating=5
        )
        logger.info("Content recommender initialized successfully")
    def _init_hybrid_rec(self):
        self._hybrid_rec = HybridRecommender.get_instance(
            lightgcn_recommender=self._lightgcn_rec,
            content_recommender=self._content_rec,
            alpha_max=self._config['hybrid']['alpha_max'],
            alpha_min=self._config['hybrid']['alpha_min'],
        )
        logger.info("Hybrid recommender initialized successfully")

    def start_periodic_updates(self):

        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._update_periodically())
            logger.info(f"Started periodic updates with interval {self._update_interval} seconds")
        else:
            logger.warning("Periodic updates already running")
    
    async def stop_periodic_updates(self):
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:

                # wait for the task to be cancelled
                await self._update_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped periodic updates")
    
    async def _update_periodically(self):
        try:
            await asyncio.sleep(5)
            
            while True:
                try:
                    logger.info("Running scheduled update of ratings and embeddings")
                    # run the update in a separate thread to avoid blocking the event loop

                    await asyncio.to_thread(self._perform_update)
                    
                    logger.info("Completed scheduled update")
                except Exception as e:
                    logger.error(f"Error in periodic update: {str(e)}")
                
                # sleep until next update
                await asyncio.sleep(self._update_interval)
        except asyncio.CancelledError:
            logger.info("Periodic updates task cancelled")
            raise
            
    def _perform_update(self):
        # create a fresh session for each update
        db = SessionLocal()
        try:
            # create a new rating repo with a fresh session
            update_rating_repo = SQLAlchemyRatingRepo(db)
            
            # get new ratings using this temporary session
            new_ratings = update_rating_repo.get_all_positive_ratings(self._rating_threshold)
            
            # update recommender with new ratings
            if new_ratings:
                self._update_lightgcn(new_ratings)
                self._ratings = new_ratings
                logger.info(f"Updated with {len(new_ratings)} ratings")
            else:
                logger.info("No positive ratings found in the database")
        finally:
            db.close()

    def _update_lightgcn(self, new_ratings: list[Rating]):
        try:
            if not new_ratings:
                logger.warning("No ratings provided for LightGCN update")
                return

            self._lightgcn_rec.update_final_embs(new_ratings)
            logger.info("LightGCN embeddings updated successfully")
        except Exception as e:
            logger.error(f"Failed to update LightGCN embeddings: {str(e)}")
            raise RecommenderException(f"Failed to update LightGCN embeddings: {str(e)}")

    def reload_lightgcn_model(self, model_dir: str):
        try:
            if not model_dir or not os.path.exists(model_dir):
                raise InvalidRequestException(f"Invalid model directory: {model_dir}")

            self._lightgcn_rec.load_new_model(model_dir, self._ratings)
            logger.info(f"New model loaded successfully {model_dir}")
        except Exception as e:
            logger.error(f"Failed to reload LightGCN model: {str(e)}")
            raise RecommenderException(f"Failed to reload LightGCN model: {str(e)}")


    def get_user_lightgcn_recommendations(self, user_id: int, num_recs: int):
        try:
            if not isinstance(user_id, int) or user_id <= 0:
                raise InvalidRequestException(f"Invalid user ID: {user_id}")

            if not isinstance(num_recs, int) or num_recs <= 0:
                raise InvalidRequestException(f"Invalid number of recommendations: {num_recs}")
            
            if not self._lightgcn_rec.is_user_in_training_data(user_id):
                logger.warning(f"User {user_id} not in training data")
                return []

            # get the items the user has liked
            user_liked_items_ids = self._get_user_liked_item_ids(user_id, self._rating_threshold)

            # get the recommendations
            recs = self._lightgcn_rec.recommend(user_id, num_recs, user_liked_items_ids)

            # enrich
            enriched_recs = self._enrich_recs(recs)
            return enriched_recs
        except Exception as e:
            logger.error(f"Failed to get LightGCN recommendations for user {user_id}: {str(e)}")
            raise RecommendationFailedException(f"Failed to get LightGCN recommendations: {str(e)}")

    def get_user_content_recommendations(self, user_id: int, num_recs: int):
        try:
            if not isinstance(user_id, int) or user_id <= 0:
                raise InvalidRequestException(f"Invalid user ID: {user_id}")

            if not isinstance(num_recs, int) or num_recs <= 0:
                raise InvalidRequestException(f"Invalid number of recommendations: {num_recs}")

            if not self._check_if_user_exists(user_id):
                logger.warning(f"User {user_id} not found in ratings data")
                return []

            # get all user positive interactions
            user_positive_interactions = self._get_user_interactions(user_id, self._rating_threshold)

            if not user_positive_interactions:
                logger.warning(f"User {user_id} has no positive ratings")
                return []
            
            # get the items that the user has already rated
            exclude_item_ids = self._extract_item_ids(user_positive_interactions)
            
            # get the recommendations by making sure to exclude the items the user has already rated
            recs = self._content_rec.recommend(user_positive_interactions, num_recs, exclude_item_ids)

            # enrich recommendations with movie details
            enriched_recs = self._enrich_recs(recs)
            return enriched_recs
        
        except Exception as e:
            logger.error(f"Failed to get content recommendations for user {user_id}: {str(e)}")
            raise RecommendationFailedException(f"Failed to get content recommendations: {str(e)}")

    def get_user_hybrid_recommendations(self, user_id: int, num_recs: int):
        try:
            if not isinstance(user_id, int) or user_id <= 0:
                raise InvalidRequestException(f"Invalid user ID: {user_id}")

            if not isinstance(num_recs, int) or num_recs <= 0 or num_recs > 100:
                raise InvalidRequestException(f"Invalid number of recommendations: {num_recs}")

            if not self._check_if_user_exists(user_id):
                logger.warning(f"User {user_id} not found in ratings data")
                return []
            user_positive_interactions = self._get_user_interactions(user_id, self._rating_threshold)

            if not user_positive_interactions:
                logger.warning(f"User {user_id} has no positive interactions")
                return []

            # get the items the user has liked
            exclude_items = self._extract_item_ids(user_positive_interactions)
            
            # get recommendations from hybrid recommender
            recs = self._hybrid_rec.recommend(
                user_id=user_id,
                user_liked_items_ratings=user_positive_interactions,
                exclude_items=exclude_items,
                top_k=num_recs
            )

            # enrich recommendations with movie details
            enriched_recs = self._enrich_recs(recs)
            return enriched_recs
        except Exception as e:
            logger.error(f"Failed to get hybrid recommendations for user {user_id}: {str(e)}")
            raise RecommendationFailedException(f"Failed to get hybrid recommendations: {str(e)}")
        
    # used to enrich recommendations with movie details
    def _enrich_recs(self, recs: list[tuple[int, float]]) -> list[dict]:
        if not recs:
                return []

        enriched_recs = []
        for movie_id, score in recs:
            try:
                movie = self._movie_repo.get_by_id(int(movie_id))
                if movie:
                    enriched_recs.append({
                            "id": int(movie_id),
                            "score": score,
                            "title": movie.title,
                            "tmdb_id": movie.tmdb_id
                        })
                else:
                    logger.warning(f"Movie {movie_id} not found in database")
            except Exception as e:
                logger.error(f"Error enriching movie {movie_id}: {str(e)}")
                continue

        return enriched_recs
    
    # used to check if the user exists in the current ratings
    def _check_if_user_exists(self, user_id: int):
        return any(interaction.user_id == user_id for interaction in self._ratings)

    # used to extract item ids from a list of user item interactions
    def _extract_item_ids(self, user_item_interactions: list[Rating]):
        if not user_item_interactions:
            return []
        return [interaction.movie_id for interaction in user_item_interactions]


    # used to get the item ids of the items that are from interactions above the rating threshold
    def _get_user_liked_item_ids(self, user_id: int, rating_threshold: float) -> list[int]:
        user_interactions = self._get_user_interactions(user_id, rating_threshold)
        user_liked_movie_ids = [interaction.movie_id for interaction in user_interactions]
        return user_liked_movie_ids

    # used to get the user interaactions above the rating threshold
    def _get_user_interactions(self, user_id, rating_threshold: float) -> list[Rating]:
        return [interaction for interaction in self._ratings if interaction.user_id == user_id and interaction.rating >= rating_threshold]
        


    # used to calculate the interaction counts for the hybrid recommender
    def _calculate_interaction_counts(self, ratings: list[Rating]) -> dict[int, int]:
        if not ratings:
                return {}

        interaction_counts = {}
        for rating in ratings:
            movie_id = rating.movie_id
            interaction_counts[movie_id] = interaction_counts.get(movie_id, 0) + 1

        return interaction_counts


    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        try:
            if not isinstance(item_id, int) or item_id <= 0:
                raise InvalidRequestException(f"Invalid item ID: {item_id}")

            if not isinstance(top_k, int) or top_k <= 0:
                raise InvalidRequestException(f"Invalid number of similar items: {top_k}")

            if not self._movie_repo.get_by_id(item_id):
                raise ResourceNotFoundException(f"Item {item_id} not found")


            similar_items = self._content_rec.get_similar_items(item_id, top_k)
            enriched_recs = self._enrich_recs(similar_items)
            return enriched_recs
        except Exception as e:
            logger.error(f"Failed to get similar items for item {item_id}: {str(e)}")
            raise RecommendationFailedException(f"Failed to get similar items: {str(e)}")

    def generate_similar_items(self, item_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        if not item_ids:
            return []
        if len(item_ids) > 5:
            raise InvalidRequestException(f"Too many items to get similar items for: {len(item_ids)}")
        similar_items = self._content_rec.get_similar_items_to_several_items(item_ids, top_k)
        enriched_recs = self._enrich_recs(similar_items)
        return enriched_recs
