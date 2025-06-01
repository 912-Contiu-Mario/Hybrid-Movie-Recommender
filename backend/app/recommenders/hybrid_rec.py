# this will be able to combine content_rec and lightgcn_rec
# it will have the following:
# be able to recommend for users, get hybrid score, calculate alpha based on popularity
import time
from typing import List, Tuple, Dict
import logging

from app.domain.models import Rating


from app.recommenders.content_rec import ContentRecommender
from app.recommenders.lightgcn_rec import LightGCNRecommender
from app.exceptions.recommender import (
    RecommendationFailedException,
    ConfigurationException
)

logger = logging.getLogger(__name__)

class HybridRecommender:
    _instance = None

    @classmethod
    def get_instance(cls, lightgcn_recommender=None, content_recommender=None, **kwargs):
        if cls._instance is None:
            if lightgcn_recommender is None or content_recommender is None:
                raise ConfigurationException("Both recommenders must be provided for first initialization")
            cls._instance = cls(lightgcn_recommender, content_recommender, **kwargs)
            logger.info("HybridRecommender instance created")
        return cls._instance

    def __init__(
            self,
            lightgcn_recommender: LightGCNRecommender,
            content_recommender: ContentRecommender,
            alpha_max: float = 0.9,
            alpha_min: float = 0.1,
    ):
        
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ConfigurationException("alpha_min and alpha_max must be between 0 and 1, and alpha_min must be less than or equal to alpha_max")

        self.content_rec = content_recommender
        self.lightgcn_rec = lightgcn_recommender
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        
        logger.info(f"HybridRecommender initialized with alpha_max={alpha_max}, alpha_min={alpha_min}")

    # this calculated the value of alpha(content strength) for a certain item.
    # if the item is cold, it will return the min value of alpha, so it's "penalized" for not having collab score in order to only surface very similar items
    # if the items is warm, it will return the max value of alpha because setting alpha to 0.8 improves warm item performance.
    def _calculate_alpha(self, user_id: int, item_id: int) -> float:
        # warm scenario = item and user have been seen by lightgcn

        # cold scenario = item or user have not been seen by lightgcn
        # if the item is in the training data, it will be given a higher alpha value
        if self.lightgcn_rec.is_item_in_training_data(item_id) and self.lightgcn_rec.is_user_in_training_data(user_id):
            alpha = self.alpha_max
        else:
            alpha = self.alpha_min  
        return alpha
    
    # def _calculate_alpha(self, item_id: int) -> float:
    #     # warm scenario = item and user have been seen by lightgcn

    #     # cold scenario = item or user have not been seen by lightgcn
    #     # if the item is in the training data, it will be given a higher alpha value
    #     if self.lightgcn_rec.is_item_in_training_data(item_id):
    #         alpha = self.alpha_max
    #     else:
    #         alpha = self.alpha_min  
    #     return alpha

    #can be further optimized by fetching top k from both recs and then merging them
    #maybe should work due to the fact that items that have both low content and collab scores will probably not surface anyway in top k
    #high collab + low content = possible to surface
    #high collab + high content = possible to surface
    #low collab + high content = possible to surface
    #low collab + low content = impossible to surface(in theory)
    def recommend(
            self,
            user_id: int,
            user_liked_items_ratings: List[Rating],
            exclude_items: List[int] = None,
            top_k: int = 10,
    ) -> List[Tuple[int, float]]:

        if exclude_items is None:
            exclude_items = []

        if not user_liked_items_ratings:
            logger.warning(f"User {user_id} has no liked items")
            return []

        try:

            num_candidates_lightgcn = int(0.1 * self.lightgcn_rec._num_items)
            num_candidates_content = int(0.1 * len(self.content_rec._movie_ids))

            # get top k candidates from each recommender
            # we get more candidates than needed to ensure we have enough after filtering and scoring
            content_candidates = self.content_rec.recommend(user_liked_items_ratings, 1000, exclude_items)
            collab_candidates = self.lightgcn_rec.recommend(user_id, 1000, exclude_items)

            # combine unique candidates, excluding items the user has already interacted with
            candidate_ids = set()
            for item_id, _ in content_candidates + collab_candidates:
                if item_id not in exclude_items:
                    candidate_ids.add(item_id)
            
            # create maps for quick score lookup
            content_scores = {item_id: score for item_id, score in content_candidates}
            collab_scores = {item_id: score for item_id, score in collab_candidates}

            # collect scores for candidates
            raw_list = []
            for item_id in candidate_ids:
                # get scores from the maps, default to None if not found
                c = content_scores.get(item_id, None)
                cf = collab_scores.get(item_id, None)
                raw_list.append((item_id, c, cf))

            if not raw_list:
                logger.warning("No valid candidates found after scoring")
                return []

            # extract mins & maxes for non-None values
            c_vals = [r[1] for r in raw_list if r[1] is not None]
            cf_vals = [r[2] for r in raw_list if r[2] is not None]

            # Handle case where we might not have any valid scores
            if not c_vals and not cf_vals:
                logger.warning("No valid scores found from recommenders")
                return []

            # Get min and max values, defaulting to 0 if no valid values
            c_min = min(c_vals) if c_vals else 0
            c_max = max(c_vals) if c_vals else 0
            cf_min = min(cf_vals) if cf_vals else 0
            cf_max = max(cf_vals) if cf_vals else 0

            # normalize, blend, collect
            # the higher the candidate set, the more confident we can be in the normalization
            hybrid_list = []
            for item_id, c_raw, cf_raw in raw_list:

                # min‐max normalization with None handling
                if c_raw is not None and c_max > c_min:
                    c_norm = (c_raw - c_min) / (c_max - c_min)
                else:
                    c_norm = 0.0

                if cf_raw is not None and cf_max > cf_min:
                    cf_norm = (cf_raw - cf_min) / (cf_max - cf_min)
                else:
                    cf_norm = 0.0

                # get item alpha
                alpha = self._calculate_alpha(user_id, item_id)

                # combine scores
                score = alpha * cf_norm + (1.0 - alpha) * c_norm
                hybrid_list.append((item_id, score))

            # pick top‑k
            hybrid_list.sort(key=lambda x: x[1], reverse=True)
            recommendations = hybrid_list[:top_k]
            return recommendations
        except Exception as e:
            logger.error(f"Failed to generate hybrid recommendations: {str(e)}")
            raise RecommendationFailedException(f"Failed to generate hybrid recommendations: {str(e)}")
        

    
    # #can be further optimized by fetching top k from both recs and then merging them
    # #maybe should work due to the fact that items that have both low content and collab scores will probably not surface anyway in top k
    # #high collab + low content = possible to surface
    # #high collab + high content = possible to surface
    # #low collab + high content = possible to surface
    # #low collab + low content = impossible to surface(in theory)
    # def recommend(
    #         self,
    #         user_id: int,
    #         user_liked_items_ratings: List[Rating],
    #         exclude_items: List[int] = None,
    #         top_k: int = 10,
    # ) -> List[Tuple[int, float]]:
    #     print("recommend hybrid time: ", time.time())

    #     if exclude_items is None:
    #         exclude_items = []

    #     if not user_liked_items_ratings:
    #         logger.warning(f"User {user_id} has no liked items")
    #         return []
        
    #     user_liked_items_ids = [interaction.movie_id for interaction in user_liked_items_ratings]
    #     exclude = set(user_liked_items_ids)

    #     # build exclude set: already liked + any passed‐in excludes
    #     exclude = exclude.union(exclude_items)

    #      # candidate universe
    #     all_items = self.content_rec._movie_ids

    #     raw_list = []
    #     for item_id in all_items:
    #         if item_id in exclude:
    #             continue

    #         c = self.content_rec.get_content_score(item_id, user_liked_items_ratings) or None

    #         cf = self.lightgcn_rec.get_collab_score(user_id, item_id) or None

    #         raw_list.append((item_id, c, cf))

    #     if not raw_list:
    #         logger.warning("No valid candidates found after scoring")
    #         return []
        
    #     c_vals = [r[1] for r in raw_list if r[1] is not None]
    #     cf_vals = [r[2] for r in raw_list if r[2] is not None]
    #     if not c_vals and not cf_vals:
    #         return []

    #     c_min = min(c_vals) if c_vals else 0
    #     c_max = max(c_vals) if c_vals else 0
    #     cf_min = min(cf_vals) if cf_vals else 0
    #     cf_max = max(cf_vals) if cf_vals else 0

    #     # normalize, blend, collect
    #     hybrid_list = []
    #     for item_id, c_raw, cf_raw in raw_list:
    #         # min‐max normalization with None handling
    #         if c_raw is not None and c_max > c_min:
    #             c_norm = (c_raw - c_min) / (c_max - c_min)
    #         else:
    #             c_norm = 0.0

    #         if cf_raw is not None and cf_max > cf_min:
    #             cf_norm = (cf_raw - cf_min) / (cf_max - cf_min)
    #         else:
    #             cf_norm = 0.0

    #         # get item alpha
    #         alpha = self._calculate_alpha(item_id)

    #         # combine scores
    #         score = alpha * cf_norm + (1.0 - alpha) * c_norm
    #         hybrid_list.append((item_id, score))

    #     # pick top‑k
    #     hybrid_list.sort(key=lambda x: x[1], reverse=True)
    #     return hybrid_list[:top_k]

        

    # def recommend(
    #         self,
    #         user_id: int,
    #         user_liked_items_ratings: List[Rating],
    #         exclude_items: List[int] = None,
    #         top_k: int = 10,
    # ) -> List[Tuple[int, float]]:

    #     if exclude_items is None:
    #         exclude_items = []

    #     if not user_liked_items_ratings:
    #         logger.warning(f"User {user_id} has no liked items")
    #         return []

    #     try:
    #         print("Generating hybrid recommendations...")
    #         # 1. candidate generation (oversample a bit)
    #         k_pool = top_k * 3
    #         content_candidates = self.content_rec.recommend(
    #             user_liked_items_ratings, k_pool, exclude_items)
    #         collab_candidates = self.lightgcn_rec.recommend(
    #             user_id, k_pool, exclude_items)

    #         # 2. build reciprocal-rank dictionaries
    #         rr_content = self._reciprocal_rank_dict(content_candidates)
    #         rr_cf      = self._reciprocal_rank_dict(collab_candidates)

    #         # 3. merge the candidate ids
    #         candidate_ids = set(rr_content) | set(rr_cf)
    #         candidate_ids -= set(exclude_items)

    #         if not candidate_ids:
    #             logger.warning("No valid candidates found")
    #             return []

    #         # 4. blend with α
    #         eps = 1e-9           # tie-breaker / smoothing
    #         blended = []
    #         for item_id in candidate_ids:
    #             r_c  = rr_content.get(item_id, 0.0)
    #             r_cf = rr_cf.get(item_id, 0.0)

    #             alpha = self._calculate_alpha(user_id, item_id)
    #             score = alpha * r_cf + (1.0 - alpha) * r_c + eps
    #             blended.append((item_id, score))

    #         # 5. rank & return
    #         blended.sort(key=lambda x: x[1], reverse=True)
    #         return blended[:top_k]

    #     except Exception as e:
    #         logger.error(f"Failed to generate hybrid recommendations: {e}")
    #         raise RecommendationFailedException(
    #             f"Failed to generate hybrid recommendations: {e}")
        
    # @staticmethod
    # def _reciprocal_rank_dict(candidates: List[Tuple[int, float]]
    #                           ) -> Dict[int, float]:
    #     """
    #     Turn a list (item_id, raw_score) sorted in ANY order into
    #     {item_id: reciprocal_rank}.  Top item → 1.0, 2nd → 0.5, etc.
    #     """
    #     # sort descending by score first
    #     ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
    #     return {item_id: 1.0 / (idx + 1)          # idx starts at 0
    #             for idx, (item_id, _) in enumerate(ordered)}
