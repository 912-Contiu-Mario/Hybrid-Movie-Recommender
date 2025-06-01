#after the model is trained, the following is necessary
#load the model, along with the mappings, so we know the real user/item ids
#also would be good if the edge index could be loaded(it's necesarry for getting the final embs by running forward pass
#final embs must be saved somewhere, so we do need the edge index to get them
#lighcn recommender must have the following functions:
#recommend for user
#get collab score for item and user
#(optional) update
import time
import os
import pickle
from typing import Optional, Tuple, List
import logging
from readerwriterlock import rwlock

import numpy as np
import torch

from app.domain.models import Rating

from app.models.lightgcn import LightGCN
from app.exceptions.recommender import (
    ModelNotLoadedException,
    RecommenderException,
    ResourceNotFoundException,
    InvalidRequestException,
    RecommendationFailedException,
    ConfigurationException
)

logger = logging.getLogger(__name__)

class LightGCNRecommender:
    _instance = None

    @classmethod
    def get_instance(cls, model_dir=None, ratings=None, device='cpu'):
        if cls._instance is None:
            if model_dir is None:
                raise ConfigurationException("model_dir must be provided for first initialization")
            if ratings is None:
                raise ConfigurationException("ratings must be provided for first initialization")
            
            cls._instance = cls(model_dir, ratings, device)

            logger.info("LightGCNRecommender instance created")
        return cls._instance

    def __init__(self, model_dir: str, ratings: list[Rating], device: str = 'cpu'):
        
        self._model = None

        try:
            self._device = torch.device(device)
        except Exception as e:
            raise ConfigurationException(f"Invalid device configuration: {str(e)}")

        self._num_users = None
        self._num_items = None

        self._user_id_to_idx = None
        self._item_id_to_idx = None
        self._idx_to_user_id = None
        self._idx_to_item_id = None

        self._final_user_embs = None
        self._final_item_embs = None

        self._embedding_dim = None
        self._n_layers = None

        self._adj_edge_index = None

        self._version = None

        # reader writer lock will allow for multiple readers on a single lock, but only one writer
        self._update_lock = rwlock.RWLockWrite()

        self.load_new_model(model_dir, ratings)
        logger.info(f"LightGCNRecommender initialized with device={device}")


    def load_new_model(self, model_dir: str, new_ratings: list[Rating]) -> None:
        logger.info(f"Loading new model from {model_dir}")

        if self._version == model_dir:
            logger.info("New model is the same as the current model")
            return
        
        model_path = os.path.join(model_dir, "model.pt")
        data_path = os.path.join(model_dir, "lightgcn_data.pkl")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            # extract the user and item mappings from the training data
            user_mapping = data['user_mapping']
            item_mapping = data['item_mapping']

            # create the mappings for the user and item ids
            idx_to_user_id = {idx: user_id for idx, user_id in enumerate(user_mapping)}
            idx_to_item_id = {idx: item_id for idx, item_id in enumerate(item_mapping)}
            user_id_to_idx = {user_id: idx for idx, user_id in idx_to_user_id.items()}
            item_id_to_idx = {item_id: idx for idx, item_id in idx_to_item_id.items()}

            # extract the config used for training the model
            config = data['config']
            embedding_dim = config.get('embedding_dim', 64)
            n_layers = config.get('n_layers', 3)

            # extract the number of users and items from the mappings
            num_users = len(user_mapping)
            num_items = len(item_mapping)

            # create the model
            new_model = LightGCN(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                n_layers=n_layers
            ).to(self._device)

            # load the model
            new_model.load_state_dict(torch.load(model_path, map_location=self._device))
            new_model.eval()

            # check if there are any new ratings
            if not new_ratings:
                logger.warning("No ratings provided for edge index construction")
                raise ConfigurationException("No ratings provided for edge index construction")

            
            # construct the edge index for forward pass
            adj_edge_index = self._construct_edge_index(
                ratings=new_ratings,
                user_id_to_idx=user_id_to_idx,
                item_id_to_idx=item_id_to_idx,
                num_users=num_users,
                device=self._device
            )

            if adj_edge_index is None:
                logger.warning("No valid ratings found for edge index construction")
                raise ConfigurationException("No valid ratings found for edge index construction")

            # generate the final embeddings using the edge index
            with torch.no_grad():
                users_emb_final, _, items_emb_final, _ = new_model(adj_edge_index)
                final_user_embs = users_emb_final.cpu().numpy()
                final_item_embs = items_emb_final.cpu().numpy()
            
            # update the state of the recommender with lock to ensure state consistency
            with self._update_lock.gen_wlock():
                self._model = new_model
                self._embedding_dim = embedding_dim
                self._n_layers = n_layers
                self._num_users = num_users
                self._num_items = num_items
                self._idx_to_user_id = idx_to_user_id
                self._idx_to_item_id = idx_to_item_id
                self._user_id_to_idx = user_id_to_idx
                self._item_id_to_idx = item_id_to_idx
                self._adj_edge_index = adj_edge_index
                self._final_user_embs = final_user_embs
                self._final_item_embs = final_item_embs
                self._version = model_dir
            
            logger.info("Loaded new model successfully")
            
        except Exception as e:
            logger.error(f"Failed to load new model: {str(e)}")
            raise RecommenderException(f"Failed to load new model: {str(e)}")
        

    def update_final_embs(self, new_ratings: list[Rating]) -> None:
        # construct the edge index for forward pass
        new_edge_index = self._construct_edge_index(
            ratings=new_ratings,
            user_id_to_idx=self._user_id_to_idx,
            item_id_to_idx=self._item_id_to_idx,
            num_users=self._num_users,
            device=self._device
        )

        if new_edge_index is None:
            logger.warning("No valid ratings found for edge index construction")
            raise RecommenderException("No valid ratings found for edge index construction")

        # generate the final embeddings using the edge index
        self._model.eval()
        with torch.no_grad():
            users_emb_final, _, items_emb_final, _ = self._model(new_edge_index)
            
        final_user_embs = users_emb_final.cpu().numpy()
        final_item_embs = items_emb_final.cpu().numpy()

        with self._update_lock.gen_wlock():
            self._adj_edge_index = new_edge_index
            self._final_user_embs = final_user_embs
            self._final_item_embs = final_item_embs


    def _construct_edge_index(
        self,
        ratings: list[Rating],
        user_id_to_idx: dict[int, int],
        item_id_to_idx: dict[int, int],
        num_users: int,
        device: str
    ) -> torch.Tensor:
        if not ratings:
            logger.warning("No ratings provided for edge index construction")
            return None
        
        # Extract IDs as numpy arrays for faster processing
        user_ids = np.array([rating.user_id for rating in ratings])
        movie_ids = np.array([rating.movie_id for rating in ratings])
        
        # Create sets of valid IDs for faster lookup
        valid_users = set(user_id_to_idx.keys())
        valid_items = set(item_id_to_idx.keys())
        
        # Create masks for valid entries
        user_mask = np.isin(user_ids, list(valid_users))
        item_mask = np.isin(movie_ids, list(valid_items))
        valid_mask = user_mask & item_mask
        
        # Filter valid entries
        valid_user_ids = user_ids[valid_mask]
        valid_movie_ids = movie_ids[valid_mask]
        
        if len(valid_user_ids) == 0:
            logger.warning("No valid ratings found for edge index construction")
            return None
        
        # Map to indices
        user_indices = np.array([user_id_to_idx[uid] for uid in valid_user_ids])
        item_indices = np.array([item_id_to_idx[mid] for mid in valid_movie_ids])
        
        # Create edge index tensor
        interaction_edge_index = torch.tensor(
            [user_indices.tolist(), item_indices.tolist()], 
            dtype=torch.long
        )
        
        # Convert to adjacency matrix representation
        adj_edge_index = self._convert_r_mat_to_adj_mat(
            interaction_edge_index, num_users
        ).to(device)
        
        return adj_edge_index



    # this is used to convert from interaction matrix edge index to adjacency matrix edge index, necessary for bipartite graphs
    def _convert_r_mat_to_adj_mat(self,
            input_edge_index,
            num_users):

        # extract user and book indices from the input edge index
        user_indices = input_edge_index[0]
        book_indices = input_edge_index[1]

        # for the top right block (users to books), book indices need to be shifted by num_users
        edge_index_top_right = torch.stack([user_indices, book_indices + num_users], dim=0)

        # For the bottom left block (books to users), similarly shift book indices.
        edge_index_bottom_left = torch.stack([book_indices + num_users, user_indices], dim=0)

        # concatenate both parts to get the full sparse edge index for the bipartite graph.
        adj_edge_index = torch.cat([edge_index_top_right, edge_index_bottom_left], dim=1)

        return adj_edge_index
    
    def is_user_in_training_data(self, user_id: int) -> bool:
        with self._update_lock.gen_rlock():
            if self._user_id_to_idx is None:
                logger.error("User mappings not loaded")
                raise ModelNotLoadedException("LightGCNRecommender")
            return user_id in self._user_id_to_idx
        
    def is_item_in_training_data(self, item_id: int) -> bool:
        with self._update_lock.gen_rlock():
            if self._item_id_to_idx is None:
                logger.error("Item mappings not loaded")
                raise ModelNotLoadedException("LightGCNRecommender")
            return item_id in self._item_id_to_idx



    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        with self._update_lock.gen_rlock():
            if self._final_user_embs is None:
                logger.error("User embeddings not generated")
                raise ModelNotLoadedException("LightGCNRecommender")
                
            if user_id not in self._user_id_to_idx:
                # logger.warning(f"User {user_id} not found in embeddings")
                # raise ResourceNotFoundException("user", user_id)
                return None
                
            user_idx = self._user_id_to_idx[user_id]
            return self._final_user_embs[user_idx]



    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        with self._update_lock.gen_rlock():
            if self._final_item_embs is None:
                logger.error("Item embeddings not generated")
                raise ModelNotLoadedException("LightGCNRecommender")
            
            if item_id not in self._item_id_to_idx:
                # logger.warning(f"Item {item_id} not found in embeddings")
                return None
                # raise ResourceNotFoundException("item", item_id)
            
            item_idx = self._item_id_to_idx[item_id]
            return self._final_item_embs[item_idx]


    #maybe collab score should be normalized here, not in the hybrid model.
    #in this case, dot product is used, which is standard
    def get_collab_score(self, user_id: int, item_id: int) -> Optional[float]:
        try:
            user_emb = self.get_user_embedding(user_id)
            item_emb = self.get_item_embedding(item_id)

            if user_emb is None or item_emb is None:
                return None


            score = np.dot(user_emb, item_emb)
            return float(score)
        except Exception as e:
            logger.error(f"Failed to calculate collaborative score for user {user_id} and item {item_id}: {str(e)}")
            return None

    def recommend(self,
                  user_id: int,
                  top_k: int = 10,
                  exclude_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        
        
        with self._update_lock.gen_rlock():
            if self._final_user_embs is None or self._final_item_embs is None:
                logger.error("Embeddings not generated")
                raise ModelNotLoadedException("LightGCNRecommender")
            
            if top_k <= 0:
                logger.error(f"Invalid top_k value: {top_k}")
                raise InvalidRequestException(f"top_k must be positive, got {top_k}")
            
            if user_id not in self._user_id_to_idx:
                # logger.warning(f"User {user_id} not found in embeddings")
                return []
            
            try:
                user_idx = self._user_id_to_idx[user_id]
                user_emb = self._final_user_embs[user_idx]
                exclude_set = set() if exclude_items is None else set(exclude_items)
                scores = []

                for item_idx in range(self._num_items):
                    item_id = self._idx_to_item_id[item_idx]
                    if item_id in exclude_set:
                        continue
                    score = np.dot(user_emb, self._final_item_embs[item_idx])
                    scores.append((item_id, float(score)))

                scores.sort(key=lambda x: x[1], reverse=True)
                recommendations = scores[:top_k]

                # make sure the ids are ints
                recommendations = [(int(item_id), float(score)) for item_id, score in recommendations]
                return recommendations
            
            except Exception as e:
                logger.error(f"Failed to generate recommendations: {str(e)}")
                raise RecommendationFailedException(f"Failed to generate recommendations: {str(e)}")
