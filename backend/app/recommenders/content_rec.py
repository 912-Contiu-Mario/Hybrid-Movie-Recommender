# must be able to load the item embs
# it has to have the following functions:
# recommend for user
# get content score for user and item
import time
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional
import os
import logging
from readerwriterlock import rwlock

from app.domain.models import Rating

from app.exceptions.recommender import (
    ModelNotLoadedException,
    ResourceNotFoundException,
    InvalidRequestException,
    RecommendationFailedException,
    ConfigurationException
)

logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ContentRecommender:
    _instance = None

    @classmethod
    def get_instance(cls, embeddings_path=None, **kwargs):
        if cls._instance is None:
            if embeddings_path is None:
                raise ConfigurationException("embeddings_path must be provided for first initialization")
            cls._instance = cls(embeddings_path, **kwargs)
            logger.info("ContentRecommender instance created")
        return cls._instance

    def __init__(self,
                 embeddings_path: str,
                 min_rating: float = 3.5,
                 min_weight: float = 0.7,
                 max_rating: float = 5,
                faiss_top_n_per_liked_item: int = 50
                 ):
        
        if min_rating >= max_rating:
            raise ConfigurationException("min_rating must be less than max_rating")
        if not 0 <= min_weight <= 1:
            raise ConfigurationException("min_weight must be between 0 and 1")

        self._embeddings = None
        self._movie_ids = None
        self._movie_idx_map = None
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_weight = min_weight
        self.faiss_top_n_per_liked_item = faiss_top_n_per_liked_item # Store this

        self._faiss_index = None

        # reader writer lock will allow for multiple readers on a single lock, but only one writer
        self._update_lock = rwlock.RWLockWrite()

        self._load_embeddings(embeddings_path)
        logger.info(f"ContentRecommender initialized with min_rating={min_rating}, min_weight={min_weight}, max_rating={max_rating}")

    def update_embs(self, embeddings_path: str)-> None:
        logger.info(f"Updating embeddings from {embeddings_path}")
        self._load_embeddings(embeddings_path)

    def _load_embeddings(self, embeddings_path: str) -> None:
        if not os.path.exists(embeddings_path):
            logger.error(f"Embeddings file not found at {embeddings_path}")
            raise ResourceNotFoundException(f"Embeddings file not found at {embeddings_path}")
        
        try:
            with open(embeddings_path, 'rb') as f:
                combined_data = pickle.load(f)

            new_embs = combined_data['embeddings']
            new_ids = combined_data['movieId']
            new_map = {movie_id: idx for idx, movie_id in enumerate(new_ids)}

            dim = new_embs.shape[1]
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(new_embs.astype(np.float32))

            # use writer lock for updating the properties, this will ensure that the properties are always in sync
            with self._update_lock.gen_wlock():
                self._embeddings = new_embs
                self._movie_ids = new_ids
                self._movie_idx_map = new_map
                self._faiss_index = new_index
            logger.info(f"Loaded embeddings for {len(self._movie_ids)} movies")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise RecommendationFailedException(f"Failed to load embeddings: {str(e)}")

    # thread safe
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        # multiple readers can read embeddings simultaneously
        # writer blocks readers
        with self._update_lock.gen_rlock():
            if self._embeddings is None:
                logger.error("Embeddings not loaded")
                raise ModelNotLoadedException("ContentRecommender")
            
            if item_id not in self._movie_idx_map:
                logger.warning(f"Item {item_id} not found in embeddings")
                raise ResourceNotFoundException("item", item_id)
            
            idx = self._movie_idx_map[item_id]
            return self._embeddings[idx]

    # returns base similarity between two item embeddings, this is the cosine similarity because the embeddings are normalized
    # thread safe
    def get_item_similarity(self, item_id1: int, item_id2: int) -> Optional[float]:
        try:
            emb1 = self.get_item_embedding(item_id1)
            emb2 = self.get_item_embedding(item_id2)

            if emb1 is None or emb2 is None:
                return None

            similarity = np.dot(emb1, emb2)

            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity between items {item_id1} and {item_id2}: {str(e)}")
            return None

    # calculate the rating weight based on the rating of the item, this is in order to give higher rated items more weight
    def _calculate_rating_weight(self, rating: float) -> float:
        scale = (rating - self.min_rating) / (self.max_rating - self.min_rating)
        return self.min_weight + scale * (1 - self.min_weight)

    # max similarity between item and user's liked items
    def get_content_score(self, movie_id: int, liked_items_ratings: list[Rating]) -> Optional[float]:
        if not liked_items_ratings:
            logger.debug("No liked items provided for content score calculation")
            return None

        try:
            # get all embeddings at once
            with self._update_lock.gen_rlock():
                if self._embeddings is None:
                    logger.error("Embeddings not loaded")
                    raise ModelNotLoadedException("ContentRecommender")
                
                # get target item embedding
                if movie_id not in self._movie_idx_map:
                    logger.warning(f"Item {movie_id} not found in embeddings")
                    return None
                item_emb = self._embeddings[self._movie_idx_map[movie_id]]

                # get all liked item embeddings at once
                liked_item_embs = {}
                for liked_item in liked_items_ratings:
                    liked_item_id = liked_item.movie_id
                    if liked_item_id == movie_id:
                        continue
                    if liked_item_id not in self._movie_idx_map:
                        continue
                    liked_item_embs[liked_item_id] = self._embeddings[self._movie_idx_map[liked_item_id]]

            max_weighted_similarity = 0.0
            for liked_item in liked_items_ratings:
                liked_item_id = liked_item.movie_id
                liked_item_rating = liked_item.rating
                
                if liked_item_id not in liked_item_embs:
                    continue

                # dot product since embeddings are normalized
                similarity = np.dot(item_emb, liked_item_embs[liked_item_id])
                rating_weight = self._calculate_rating_weight(liked_item_rating)
                weighted_similarity = similarity * rating_weight
                max_weighted_similarity = max(max_weighted_similarity, weighted_similarity)

            return max_weighted_similarity
        except Exception as e:
            logger.error(f"Failed to calculate content score for movie {movie_id}: {str(e)}")
            return None

    # search similar items for each liked item, merge scores, return top k
   # search similar items for each liked item, merge scores, return top k
    # OPTIMIZED recommend method
    def recommend(
            self,
            user_liked_items: List[Rating],
            top_k: int = 10,
            exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        if user_liked_items is None: # Explicit check for None
            raise InvalidRequestException("User liked items not provided.")
        
        if not user_liked_items:
            return []

        if top_k <= 0:
            raise InvalidRequestException(f"top_k must be positive, got {top_k}")

        # Prepare exclusion set: items explicitly excluded + items already liked by the user
        exclude_set = set(exclude_items) if exclude_items else set()
        for ui in user_liked_items:
            exclude_set.add(ui.movie_id)

        query_embeddings_list = []
        query_weights = []

        try:
            with self._update_lock.gen_rlock():
                if self._embeddings is None or self._faiss_index is None or self._movie_ids is None or self._movie_idx_map is None:
                    logger.error("ContentRecommender not fully loaded (embeddings, Faiss index, or movie_ids/map missing)")
                    raise ModelNotLoadedException("ContentRecommender")

                for ui in user_liked_items:
                    liked_id = ui.movie_id
                    liked_rating = ui.rating
                    
                    if liked_id not in self._movie_idx_map:
                        continue 
                        
                    query_embeddings_list.append(self._embeddings[self._movie_idx_map[liked_id]])
                    query_weights.append(self._calculate_rating_weight(liked_rating))

                if not query_embeddings_list:
                    return []

                query_vectors_np = np.array(query_embeddings_list).astype(np.float32)
                
                num_total_embeddings = len(self._movie_ids)

                num_neighbors_to_fetch = min(top_k, num_total_embeddings)
                
                if num_neighbors_to_fetch <= 0 and num_total_embeddings > 0:
                    num_neighbors_to_fetch = 1 
                
                candidate_scores = {}
                if num_neighbors_to_fetch > 0 : # Only search if k is positive
                    all_similarities, all_indices = self._faiss_index.search(query_vectors_np, num_neighbors_to_fetch)
                
                    for i in range(len(query_vectors_np)): # For each liked item query
                        weight = query_weights[i]
                        
                        for j in range(num_neighbors_to_fetch): # For each neighbor found
                            candidate_idx = all_indices[i][j]
                            if candidate_idx == -1: 
                                continue 
                            
                            candidate_id = self._movie_ids[candidate_idx]
                            
                            if candidate_id in exclude_set:
                                continue
                                
                            similarity_score = all_similarities[i][j] 
                            weighted_sim = float(similarity_score) * weight
                            
                            candidate_scores[candidate_id] = max(candidate_scores.get(candidate_id, -float('inf')), weighted_sim)

            # Sort and return top-k (outside the lock)
            if not candidate_scores:
                return []
                
            scores = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            return scores[:top_k]
        except ModelNotLoadedException: 
            raise
        except InvalidRequestException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}", exc_info=True) 
            raise RecommendationFailedException(f"Failed to generate recommendations: {str(e)}")
        
    def _create_averaged_embeddings(self, item_ids: List[int]) -> np.ndarray:
        if not item_ids:
            return None
        
        with self._update_lock.gen_rlock():
            if self._embeddings is None:
                raise ModelNotLoadedException("ContentRecommender")
            
            item_embs = []
            for item_id in item_ids:
                if item_id not in self._movie_idx_map:
                    continue
                item_embs.append(self._embeddings[self._movie_idx_map[item_id]])

            if not item_embs:
                return None

            averaged_emb = np.mean(item_embs, axis=0)
            return averaged_emb

    def get_similar_items_to_several_items(self, item_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        if not item_ids:
            return []
        
        averaged_emb = self._create_averaged_embeddings(item_ids)
        if averaged_emb is None:
            return []
        
        with self._update_lock.gen_rlock():
            if self._embeddings is None:
                raise ModelNotLoadedException("ContentRecommender")
            
            # search for the most similar items
            k = min(top_k + 1, len(self._embeddings))
            query_vector = averaged_emb.reshape(1, -1)
            similarities, indices = self._faiss_index.search(query_vector, k)

            results = []
            for i, idx in enumerate(indices[0]):
                other_id = self._movie_ids[idx] 
                if other_id in item_ids:
                    continue
                results.append((other_id, float(similarities[0][i])))

            return results

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if top_k <= 0:
            raise InvalidRequestException(f"top_k must be positive, got {top_k}")

        try:
            with self._update_lock.gen_rlock():
                if self._embeddings is None:
                    raise ModelNotLoadedException("ContentRecommender")

                if item_id not in self._movie_idx_map:
                    logger.warning(f"Item {item_id} not found in embeddings")
                    return []

                # get the item embedding
                item_emb = self._embeddings[self._movie_idx_map[item_id]]

                # search for the most similar items
                # add 1 to top_k to include the query item itself, which we'll filter out
                k = min(top_k + 1, len(self._embeddings))
                query_vector = item_emb.reshape(1, -1)
                similarities, indices = self._faiss_index.search(query_vector, k)
                
                # Get all movie IDs while still holding the lock
                results = []
                for i, idx in enumerate(indices[0]):
                    other_id = self._movie_ids[idx]
                    if other_id != item_id:
                        results.append((other_id, float(similarities[0][i])))
                        if len(results) >= top_k:
                            break

            return results

        except Exception as e:
            logger.error(f"Failed to find similar items: {str(e)}")
            raise RecommendationFailedException(f"Failed to find similar items: {str(e)}")
