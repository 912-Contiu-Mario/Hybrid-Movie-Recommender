import threading
import time
import logging
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("concurrency_test")

def setup_test_environment():
    """Set up a test environment with real components"""
    from app.db.database import SessionLocal
    from app.repositories.sqlite import SQLiteRatingRepository, SQLiteMovieRepository, SQLiteUserRepository
    from app.service.rec_service import RecService
    from app.recommenders.content_rec import ContentRecommender
    from app.recommenders.lightgcn_rec import LightGCNRecommender
    from app.recommenders.hybrid_rec import HybridRecommender
    from app.config import get_recommender_config

    # Get a database session
    db = SessionLocal()
    
    try:
        # Get config
        config = get_recommender_config()
        
        # Initialize repositories
        rating_repo = SQLiteRatingRepository(db)
        movie_repo = SQLiteMovieRepository(db)
        user_repo = SQLiteUserRepository(db)

        # Initialize recommenders
        lightgcn_rec = LightGCNRecommender.get_instance(
            model_dir=config['lightgcn']["lightgcn_model_path"],
            device="cpu"
        )

        content_rec = ContentRecommender.get_instance(
            embeddings_path=config['content']['embeddings_path'],
            min_rating=config['rating_threshold'],
            min_weight=config['content']['min_weight'],
            max_rating=5
        )

        hybrid_rec = HybridRecommender.get_instance(
            lightgcn_recommender=lightgcn_rec,
            content_recommender=content_rec,
            alpha_max=config['hybrid']['alpha_max'],
            alpha_min=config['hybrid']['alpha_min'],
        )

        # Initialize rec service
        rec_service = RecService(
            content_recommender=content_rec,
            lightgcn_recommender=lightgcn_rec,
            hybrid_recommender=hybrid_rec,
            rating_repository=rating_repo,
            user_repository=user_repo,
            movie_repository=movie_repo,
            config=config
        )
        
        # Do initial update
        rec_service.update_ratings()

        return {
            'db': db,
            'rec_service': rec_service,
            'repos': {
                'rating': rating_repo,
                'movie': movie_repo,
                'user': user_repo
            }
        }
    except Exception as e:
        db.close()
        logger.error(f"Failed to set up test environment: {str(e)}")
        raise

def slow_update_ratings(rec_service):
    """Modified update function that's slow for testing"""
    logger.info(f"[TEST] Starting slow update at {datetime.now().strftime('%H:%M:%S.%f')}")
    
    # Sleep to simulate long update
    time.sleep(5)
    logger.info(f"[TEST] Mid-update at {datetime.now().strftime('%H:%M:%S.%f')}")
    
    # Actual update
    rec_service.update_ratings()
    
    logger.info(f"[TEST] Completed update at {datetime.now().strftime('%H:%M:%S.%f')}")

def get_recommendation(rec_service, user_id, thread_id):
    """Get recommendation and log timing"""
    try:
        start_time = datetime.now()
        logger.info(f"[TEST] Thread {thread_id} starting recommendation at {start_time.strftime('%H:%M:%S.%f')}")
        
        # Get recommendations
        recommendations = rec_service.get_user_lightgcn_recommendations(user_id, 10)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"[TEST] Thread {thread_id} got {len(recommendations)} recommendations in {duration:.2f}s")
        
        return recommendations
    except Exception as e:
        logger.error(f"[TEST] Thread {thread_id} failed: {str(e)}")
        return None

def run_concurrency_test():
    """Run a direct concurrency test"""
    logger.info("Setting up test environment...")
    env = setup_test_environment()
    rec_service = env['rec_service']
    db = env['db']
    
    try:
        logger.info("Starting concurrency test...")
        
        # Create and start update thread
        update_thread = threading.Thread(
            target=slow_update_ratings, 
            args=(rec_service,)
        )
        update_thread.start()
        
        # Give update a moment to start
        time.sleep(0.5)
        
        # Create recommendation threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                # Start recommendation in thread pool
                future = executor.submit(get_recommendation, rec_service, 1, i)
                futures.append(future)
            
            # Wait for all recommendation threads to complete
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    logger.info(f"[TEST] Thread {i} result: {'Success' if result else 'Failed'}")
                except Exception as e:
                    logger.error(f"[TEST] Thread {i} exception: {str(e)}")
        
        # Wait for update thread to complete
        update_thread.join()
        
        logger.info("\n[TEST] Test completed.")
        logger.info("[TEST] If recommendation threads complete while the update is still running,")
        logger.info("[TEST] then concurrency is working correctly. If all recommendation requests")
        logger.info("[TEST] wait for the update to complete, then there may be issues with the RW locks.")
        
    finally:
        db.close()

if __name__ == "__main__":
    run_concurrency_test() 