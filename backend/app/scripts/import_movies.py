import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging

from app.config import MOVIES_PATH, RATINGS_SMALL_PATH
from app.db.database import SessionLocal
from app.db.models import MovieORM, UserORM, RatingORM, Base
from app.db.database import engine



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_movies_from_csv(csv_path: Path, batch_size: int = 100) -> tuple[int, int, int]:
    """Import movies from CSV file.
    
    Returns:
        tuple: (added_count, skipped_count, total_count)
    """
    db_session = SessionLocal()

    try:
        logger.info(f"Reading movies from {csv_path}...")
        movies = pd.read_csv(csv_path)

        initial_count = len(movies)

        # Deduplicate based on movieId
        duplicate_movieids = movies.duplicated(subset=['movieId']).sum()
        if duplicate_movieids > 0:
            logger.info(f"Found {duplicate_movieids} duplicate movie IDs in CSV, deduplicating")
            movies = movies.drop_duplicates(subset=['movieId'], keep='first')

        # Deduplicate based on tmdb id
        movies_with_tmdb = movies.dropna(subset=['tmdbId'])
        duplicate_tmdbids = movies_with_tmdb.duplicated(subset=['tmdbId']).sum()

        if duplicate_tmdbids > 0:
            logger.info(f"Found {duplicate_tmdbids} duplicate TMDB IDs, deduplicating")
            processed_tmdb_ids = set()
            movies_to_keep = []

            for _, row in movies.iterrows():
                if pd.isna(row['tmdbId']) or row['tmdbId'] not in processed_tmdb_ids:
                    movies_to_keep.append(row)
                    if not pd.isna(row['tmdbId']):
                        processed_tmdb_ids.add(row['tmdbId'])

            movies = pd.DataFrame(movies_to_keep)

        logger.info(f"After deduplication: {len(movies)} movies kept (removed {initial_count - len(movies)} duplicates)")

        # Clean data
        before_cleaning = len(movies)
        movies = movies.dropna(subset=['title', 'genres'])
        if len(movies) < before_cleaning:
            logger.info(f"Removed {before_cleaning - len(movies)} movies with missing titles or genres.")

        total = len(movies)
        added = 0
        skipped = 0
        batch_counter = 0

        logger.info(f"Processing {total} movies...")
        for index, row in tqdm(movies.iterrows(), total=total, desc="Importing movies"):
            try:
                movie_id = int(row['movieId'])

                tmdb_id = row['tmdbId']
                if pd.isna(tmdb_id):
                    tmdb_id = None
                else:
                    tmdb_id = int(tmdb_id)

                if db_session.query(MovieORM).filter(MovieORM.id == movie_id).first() is None:
                    movie = MovieORM(
                        id=movie_id,
                        tmdb_id=tmdb_id,
                        title=str(row['title']),
                        genres=str(row['genres']),
                    )

                    db_session.add(movie)
                    added += 1
                    batch_counter += 1
                else:
                    skipped += 1

                if batch_counter >= batch_size:
                    db_session.commit()
                    batch_counter = 0

            except Exception as e:
                db_session.rollback()
                logger.error(f"Error adding movie {row['movieId']}: {e}")
                batch_counter = 0

        if batch_counter > 0:
            db_session.commit()

        logger.info(f"Movie import summary: {added} added, {skipped} skipped, {total} total")
        return added, skipped, total

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error importing movies: {e}")
        raise

    finally:
        db_session.close()

def import_users_from_csv(csv_path: Path, batch_size: int = 1000) -> tuple[int, int]:
    """Import users from CSV file.
    
    Returns:
        tuple: (added_count, total_count)
    """
    db_session = SessionLocal()

    try:
        logger.info(f"Reading users from {csv_path}...")
        users = pd.read_csv(csv_path)

        duplicate_count = users.duplicated(subset=['userId']).sum()
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} duplicate user IDs in CSV. Removing duplicates...")
            users = users.drop_duplicates(subset=['userId'], keep='first')
            logger.info(f"Kept {len(users)} unique users after deduplication.")

        total = len(users)
        added = 0
        batch_counter = 0

        logger.info(f"Processing {total} users...")
        for index, row in tqdm(users.iterrows(), total=total, desc="Importing users"):
            try:
                user_id = int(row['userId'])
                
                # Generate a simple password for test users
                password = f"Admin123"

                user = UserORM(
                    id=user_id,
                    username=f"user{user_id}",
                    email=f"user{user_id}@test.com",
                    is_test=True,
                    hashed_password=password,
                    is_active=True,
                    is_admin=False
                )

                db_session.add(user)
                added += 1
                batch_counter += 1

                if batch_counter >= batch_size:
                    db_session.commit()
                    batch_counter = 0

            except Exception as e:
                db_session.rollback()
                logger.error(f"Error adding user {row['userId']}: {e}")
                batch_counter = 0

        if batch_counter > 0:
            db_session.commit()

        logger.info(f"User import summary: {added} added, {total} total")
        return added, total

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error importing users: {e}")
        raise

    finally:
        db_session.close()

def import_ratings_from_csv(csv_path: Path, batch_size: int = 5000) -> tuple[int, int]:
    """Import ratings from CSV file.
    
    Returns:
        tuple: (added_count, total_count)
    """
    db_session = SessionLocal()

    try:
        logger.info(f"Reading ratings from {csv_path}...")
        ratings = pd.read_csv(csv_path)

        duplicate_count = ratings.duplicated(subset=['userId', 'movieId']).sum()
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} duplicate ratings in CSV. Removing duplicates...")
            ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='first')
            logger.info(f"Kept {len(ratings)} unique ratings after deduplication.")

        total = len(ratings)
        added = 0
        skipped = 0
        batch_counter = 0
        ratings_buffer = []

        logger.info(f"Processing {total} ratings...")
        for index, row in tqdm(ratings.iterrows(), total=total, desc="Importing ratings"):
            try:
                user_id = int(row['userId'])
                movie_id = int(row['movieId'])
                rating_value = float(row['rating'])
                
                # Convert Unix timestamp to datetime
                timestamp = datetime.fromtimestamp(int(row['timestamp']))

                rating = RatingORM(
                    movie_id=movie_id,
                    user_id=user_id,
                    rating=rating_value,
                    timestamp=timestamp
                )
                
                # Add to buffer instead of directly to session
                ratings_buffer.append(rating)
                batch_counter += 1

                # When buffer reaches batch size, add to session and commit
                if batch_counter >= batch_size:
                    for r in ratings_buffer:
                        try:
                            db_session.add(r)
                            db_session.flush()  # Check constraints for this single record
                            added += 1
                        except Exception as e:
                            db_session.rollback()  # Rollback just this rating
                            skipped += 1
                            logger.debug(f"Skipped rating (user:{r.user_id}, movie:{r.movie_id}): {str(e)}")
                    
                    db_session.commit()  # Commit all successful records
                    ratings_buffer = []
                    batch_counter = 0

            except Exception as e:
                # This only handles errors in preparing rating objects
                skipped += 1
                logger.debug(f"Could not process rating row: {str(e)}")

        # Process any remaining ratings in buffer
        if ratings_buffer:
            for r in ratings_buffer:
                try:
                    db_session.add(r)
                    db_session.flush()  # Check constraints for this single record
                    added += 1
                except Exception as e:
                    db_session.rollback()  # Rollback just this rating
                    skipped += 1
                    logger.debug(f"Skipped rating (user:{r.user_id}, movie:{r.movie_id}): {str(e)}")
            
            db_session.commit()

        logger.info(f"Rating import summary: {added} added, {skipped} skipped, {total} total")
        return added, total

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error importing ratings: {e}")
        raise

    finally:
        db_session.close()

def purge_database():
    """Purge all data from the database."""
    logger.info("Purging database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database purged successfully")

def import_all(purge: bool = False):
    if purge:
        purge_database()
        
    import_movies_from_csv(MOVIES_PATH)
    import_users_from_csv(RATINGS_SMALL_PATH)
    import_ratings_from_csv(RATINGS_SMALL_PATH)

if __name__ == "__main__":
    import_all(purge=True)