import logging
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
import wandb
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

from app.db.database import SessionLocal, engine, Base
from app.repositories.sqlite import SQLiteRatingRepository
from app.models.lightgcn import LightGCN
from app.config import get_recommender_config
from app.config.training import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database and create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

#this is used to convert from interaction matrix edge index to adjacency matrix edge index, necessary for bipartite graphs
def convert_r_mat_to_adj_mat(
        input_edge_index,
        num_users,
        num_items):

    #extract user and book indices from the input edge index
    user_indices = input_edge_index[0]
    book_indices = input_edge_index[1]

    #for the top right block (users to books), book indices need to be shifted by num_users
    edge_index_top_right = torch.stack([user_indices, book_indices + num_users], dim=0)

    #For the bottom left block (books to users), similarly shift book indices.
    edge_index_bottom_left = torch.stack([book_indices + num_users, user_indices], dim=0)

    #concatenate both parts to get the full sparse edge index for the bipartite graph.
    adj_edge_index = torch.cat([edge_index_top_right, edge_index_bottom_left], dim=1)

    return adj_edge_index

#convert from adjacency matrix edge index into rating matrix edge index
def convert_adj_mat_to_r_mat(
        adj_edge_index,
        num_users):
    #the top right block has user-book edges:
    #user nodes have indices < num_users, and book nodes have indices >= num_users.
    mask = (adj_edge_index[0] < num_users) & (adj_edge_index[1] >= num_users)

    #extract the user indices as is.
    user_indices = adj_edge_index[0][mask]

    #for the book indices, subtract num_users to convert back to the original indexing.
    book_indices = adj_edge_index[1][mask] - num_users

    #return the rating matrix edge index.
    return torch.stack([user_indices, book_indices], dim=0)

def bpr_loss(users_emb_final, 
             users_emb_0, 
             pos_items_emb_final, 
             pos_items_emb_0, 
             neg_items_emb_final, 
             neg_items_emb_0, 
             lambda_val):
    
    # calculate prediction scores
    pos_scores = (users_emb_final * pos_items_emb_final).sum(1)
    neg_scores = (users_emb_final * neg_items_emb_final).sum(1)
    
    # calculate mf loss
    mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    
    # regularization using batch normalization

    #get batch size
    batch_size = float(users_emb_0.size(0))

    #calculate using batch normalization
    reg_loss = lambda_val * (1/2) * (
        users_emb_0.norm(2).pow(2) + 
        pos_items_emb_0.norm(2).pow(2) + 
        neg_items_emb_0.norm(2).pow(2)
    ) / batch_size
    
    return mf_loss + reg_loss

def random_negative_sampling(
    edge_index,
    num_users,
    num_items,
    batch_size
) :

    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index)

    # Ensure edge_index has the right shape
    if edge_index.shape[0] != 2:
        raise ValueError("Edge index should have shape [2, num_edges]")

    # Create a dictionary of user interactions for faster negative sampling
    user_interactions = {}
    for u, i in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if u not in user_interactions:
            user_interactions[u] = set()
        user_interactions[u].add(i)

    # sample positive interactions
    num_edges = edge_index.shape[1]
    pos_indices = torch.randint(0, num_edges, (batch_size,))

    # get the user that made the interaction
    user_ids = edge_index[0][pos_indices]

    #get the items that were interacted with
    pos_item_ids = edge_index[1][pos_indices]

    neg_item_ids = torch.zeros_like(user_ids)

    #sample negative item ids from the set of items that the user has not interacted with
    for i, user_id in enumerate(user_ids):
        user_id = user_id.item()

        #get set of items user interacted with
        interacted_items = user_interactions.get(user_id, set())

        #randomly select a random item, if the user interacted with this item, sample again.
        while True:
            neg_item = torch.randint(0, num_items, (1,)).item()
            if neg_item not in interacted_items:
                neg_item_ids[i] = neg_item
                break

    return user_ids, pos_item_ids, neg_item_ids

def get_user_positive_items(edge_index):
    user_pos_items = {}
    
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        
        if user not in user_pos_items:
            user_pos_items[user] = []
        
        user_pos_items[user].append(item)
        
    return user_pos_items

def RecallPrecision_ATk(
        groundTruth,
        r,
        k):

    
    # number of correctly predicted items per user
    num_correct_pred = torch.sum(r, dim=-1)  
    
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])
    
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def NDCGatK_r(
        groundTruth,
        r,
        k):


    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1


    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

def get_metrics(model, input_edge_index, num_users, train_edge_index, input_exclude_edge_indices, k):
    """Calculate evaluation metrics."""
    users_emb_final, _, items_emb_final, _ = model(train_edge_index)
    edge_index = convert_adj_mat_to_r_mat(input_edge_index, num_users)
    exclude_edge_indices = [convert_adj_mat_to_r_mat(exclude_edge_index, num_users)
                          for exclude_edge_index in input_exclude_edge_indices]
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    
    exclude_items_dict = {}
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        for user, items in user_pos_items.items():
            if user not in exclude_items_dict:
                exclude_items_dict[user] = set(items)
            else:
                exclude_items_dict[user].update(items)

    user_embeddings = users_emb_final[users]
    all_scores = torch.matmul(user_embeddings, items_emb_final.T)
    
    for user_idx, user_id in enumerate(users):
        if user_id.item() in exclude_items_dict:
            exclude_indices = list(exclude_items_dict[user_id.item()])
            if exclude_indices:
                all_scores[user_idx, exclude_indices] = float('-inf')
    
    _, top_k_indices = torch.topk(all_scores, k=k)
    
    ndcg = 0.0
    for user_idx, user_id in enumerate(users):
        user_id = user_id.item()
        true_items = set(test_user_pos_items[user_id])
        pred_items = set(top_k_indices[user_idx].tolist())
        user_ndcg = NDCGatK_r([list(true_items)], 
                             torch.tensor([[1.0 if item in true_items else 0.0 
                                         for item in pred_items]]), 
                             k)
        ndcg += user_ndcg
    
    ndcg /= len(users)
    return ndcg

def evaluation(model,
               num_users,
               edge_index, # adj_mat based edge index
               train_edge_index,
               exclude_edge_indices: List,  # adj_mat based exclude edge index
               k,
               lambda_val
              ):


    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(edge_index)

    r_mat_edge_index = convert_adj_mat_to_r_mat(edge_index, num_users)

    # get the set of all items that appear in the edge_index
    all_items = torch.unique(r_mat_edge_index[1]).tolist()

    # create a dictionary to track user interactions for negative sampling
    user_interactions = {}
    for u, i in zip(r_mat_edge_index[0].tolist(), r_mat_edge_index[1].tolist()):
        if u not in user_interactions:
            user_interactions[u] = set()
        user_interactions[u].add(i)

    # use all positive edges
    user_indices = r_mat_edge_index[0]
    pos_item_indices = r_mat_edge_index[1]

    # generate negative item indices with the same shape as pos_item_indices
    neg_item_indices = torch.zeros_like(pos_item_indices)

    for idx, user_id in enumerate(user_indices):
        user_id = user_id.item()
        interacted_items = user_interactions.get(user_id, set())

        # find items this user hasn't interacted with
        available_neg_items = [item for item in all_items if item not in interacted_items]

        if available_neg_items:

            # randomly select one negative item
            neg_item = random.choice(available_neg_items)
            neg_item_indices[idx] = neg_item
        else:
            # if all items have been interacted with by this user (rare case),
            # just pick a random item as negative
            neg_item = random.choice(all_items)
            neg_item_indices[idx] = neg_item

    # get embeddings for all users and items
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]


    #calculate the loss

    loss = bpr_loss(users_emb_final,
                    users_emb_0,
                    pos_items_emb_final,
                    pos_items_emb_0,
                    neg_items_emb_final,
                    neg_items_emb_0,
                    lambda_val).item()

    #calculate the metrics
    ndcg = get_metrics(model,
                        edge_index,
                                          num_users,
                                          train_edge_index,
                                          exclude_edge_indices,
                                          k)

    return loss, ndcg

def get_embs_for_bpr(model, input_edge_index, num_users, num_items, batch_size, device):

    #get the initial and final user/item embs from the model
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(input_edge_index)


    #convert to interaction form
    edge_index_to_use = convert_adj_mat_to_r_mat(input_edge_index, num_users)



    # mini batching for eval and loss calculation
    user_indices, pos_item_indices, neg_item_indices = random_negative_sampling(edge_index_to_use, num_users, num_items,batch_size)

    #move indices to device
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)
    
 
    # we need layer0 embeddings and the final embeddings (computed from 0...K layer) for BPR loss
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
   
    return users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0

def train_lightgcn(model, optimizer, scheduler, data, config):
    """Train the LightGCN model."""
    iterations = config.get('iterations', 5000)
    batch_size = config.get('batch_size', 2048)
    lambda_val = config.get('lambda_val', 1e-6)
    eval_every = config.get('eval_every', 200)
    lr_decay_every = config.get('lr_decay_every', 200)
    k = config.get('k', 20)

    device = next(model.parameters()).device
    train_adj_edge_index = data['train_adj_edge_index'].to(device)
    val_adj_edge_index = data['val_adj_edge_index'].to(device)

    history = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'val_ndcg': []
    }

    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 0.001)
    best_ndcg = 0
    patience_counter = 0

    model.train()

    for iteration in tqdm(range(iterations)):
        users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0 \
            = get_embs_for_bpr(model, train_adj_edge_index, data['num_users'], data['num_items'], batch_size, device)
        
        train_loss = bpr_loss(users_emb_final, users_emb_0, 
                             pos_items_emb_final, pos_items_emb_0, 
                             neg_items_emb_final, neg_items_emb_0, 
                             lambda_val)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

       
    
        if iteration % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_ndcg = evaluation(
                    model=model,
                    num_users=data['num_users'],
                    edge_index=val_adj_edge_index,
                    train_edge_index=train_adj_edge_index,
                    exclude_edge_indices=[train_adj_edge_index],
                    k=k,
                    lambda_val=lambda_val
                )

                if val_ndcg > best_ndcg + min_delta:
                    best_ndcg = val_ndcg
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"[Iteration {iteration}/{iterations}] "
                      f"train_loss: {train_loss.item():.5f}, val_loss: {val_loss:.5f}, "
                      f"val_ndcg@{k}: {val_ndcg:.5f}")

                history['iterations'].append(iteration)
                history['train_loss'].append(train_loss.item())
                history['val_loss'].append(val_loss)
                history['val_ndcg'].append(val_ndcg)

                if patience_counter >= patience:
                    print(f"Early stopping triggered at iteration {iteration}")
                    break

            model.train()

        if scheduler is not None and iteration % lr_decay_every == 0 and iteration > 0:
            scheduler.step()

    return history

def load_and_process_movies(rating_repo, rating_threshold, **kwargs):
    """Load and process movie ratings from database."""
    positive_interaction = rating_repo.get_all_positive_ratings(rating_threshold)
    ratings_df = pd.DataFrame(positive_interaction)
    print(f"Loaded {len(ratings_df)} ratings")

    if kwargs.get('user_min_interactions', 0) > 0:
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= kwargs['user_min_interactions']].index
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
        print(f"Filtered to {len(ratings_df)} ratings from users with at least {kwargs['user_min_interactions']} interactions")

    if kwargs.get('item_min_interactions', 0) > 0:
        item_counts = ratings_df['movieId'].value_counts()
        valid_items = item_counts[item_counts >= kwargs['item_min_interactions']].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_items)]
        print(f"Filtered to {len(ratings_df)} ratings for items with at least {kwargs['item_min_interactions']} interactions")

    if kwargs.get('user_max_interactions', 0) > 0:
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts < kwargs['user_max_interactions']].index
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
        print(f"Filtered to {len(ratings_df)} ratings from users with at most {kwargs['user_max_interactions']} interactions")

    if kwargs.get('item_max_interactions', 0) > 0:
        item_counts = ratings_df['movieId'].value_counts()
        valid_items = item_counts[item_counts < kwargs['item_max_interactions']].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_items)]
        print(f"Filtered to {len(ratings_df)} ratings for items with at most {kwargs['item_max_interactions']} interactions")

    user_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    user_encoder.fit(ratings_df['userId'])
    item_encoder.fit(ratings_df['movieId'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    print(f"Dataset has {num_users} users and {num_items} items")

    user_id_to_idx = {id_: idx for idx, id_ in enumerate(user_encoder.classes_)}
    item_id_to_idx = {id_: idx for idx, id_ in enumerate(item_encoder.classes_)}
    idx_to_user_id = {idx: id_ for id_, idx in user_id_to_idx.items()}
    idx_to_item_id = {idx: id_ for id_, idx in item_id_to_idx.items()}

    ratings_encoded = ratings_df.copy()
    ratings_encoded['user_idx'] = user_encoder.transform(ratings_df['userId'])
    ratings_encoded['item_idx'] = item_encoder.transform(ratings_df['movieId'])

    temp_size = kwargs.get('val_size', 0.1) + kwargs.get('test_size', 0.1)
    train_df, temp_df = train_test_split(ratings_encoded, test_size=temp_size, random_state=kwargs.get('random_state', 42))

    val_size_adjusted = kwargs.get('val_size', 0.1) / temp_size
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_size_adjusted), random_state=kwargs.get('random_state', 42))

    print(f"Split into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} testing interactions")

    train_edge_index = torch.LongTensor([
        train_df['user_idx'].values,
        train_df['item_idx'].values
    ])

    val_edge_index = torch.LongTensor([
        val_df['user_idx'].values,
        val_df['item_idx'].values
    ])

    test_edge_index = torch.LongTensor([
        test_df['user_idx'].values,
        test_df['item_idx'].values
    ])

    return {
        'num_users': num_users,
        'num_items': num_items,
        'train_edge_index': train_edge_index,
        'val_edge_index': val_edge_index,
        'test_edge_index': test_edge_index,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'user_id_to_idx': user_id_to_idx,
        'item_id_to_idx': item_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'idx_to_item_id': idx_to_item_id,
        'ratings_df': ratings_df,
        'ratings_encoded': ratings_encoded
    }

def prepare_lightgcn_data(rating_repo, **kwargs):
    """Prepare data for LightGCN training."""
    data = load_and_process_movies(rating_repo=rating_repo, **kwargs)
    
    data['train_adj_edge_index'] = convert_r_mat_to_adj_mat(
        data['train_edge_index'], data['num_users'], data['num_items']
    )
    data['val_adj_edge_index'] = convert_r_mat_to_adj_mat(
        data['val_edge_index'], data['num_users'], data['num_items']
    )
    data['test_adj_edge_index'] = convert_r_mat_to_adj_mat(
        data['test_edge_index'], data['num_users'], data['num_items']
    )
    
    return data

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(config: Dict[str, Any] = None):
    try:
        set_seed(42)
        init_db()
        logger.info("Database initialized")

        db = SessionLocal()
        try:
            rating_repo = SQLiteRatingRepository(db)
            logger.info("Rating repository initialized")

            if config is None:
                config = get_recommender_config()
            logger.info("Configuration loaded")

            models_dir = config['lightgcn']['lightgcn_model_path']
            os.makedirs(models_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            experiment_name = f"lightgcn-{timestamp}"
            save_dir = f"{models_dir}/{experiment_name}"
            os.makedirs(save_dir, exist_ok=True)

            wandb.init(
                project="lightgcn-recommendations",
                name=experiment_name,
                config=config,
                mode="offline"
            )

            logger.info("Preparing training data...")
            data = prepare_lightgcn_data(
                rating_repo=rating_repo,
                rating_threshold=config['train_config']['rating_threshold'],
                user_min_interactions=config['train_config']['user_min_interactions'],
                item_min_interactions=config['train_config']['item_min_interactions'],
                user_max_interactions=config['train_config']['user_max_interactions'],
                item_max_interactions=config['train_config']['item_max_interactions'],
                val_size=config['train_config']['val_size'],
                test_size=config['train_config']['test_size'],
                random_state=42
            )

            logger.info("Creating model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            model = LightGCN(
                num_users=data['num_users'],
                num_items=data['num_items'],
                embedding_dim=config['model_config']['embedding_dim'],
                num_layers=config['model_config']['n_layers'],
                add_self_loops=config['model_config']['add_self_loops']
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=config['train_config']['lr'])
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=config['train_config']['lr_decay_factor']
            )

            logger.info("Training model...")
            start_time = time.time()
            
            history = train_lightgcn(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=data,
                config=config['model_config']
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")

            logger.info("Saving model and data...")
            torch.save(model.state_dict(), f"{save_dir}/model.pt")

            with open(f"{save_dir}/lightgcn_data.pkl", 'wb') as f:
                pickle.dump({
                    'user_mapping': list(data['idx_to_user_id'].values()),
                    'item_mapping': list(data['idx_to_item_id'].values()),
                    'model_config': config['model_config'],
                    'training_config': config['train_config']
                }, f)

            history_df = pd.DataFrame(history)
            history_df.to_csv(f"{save_dir}/history.csv", index=False)

            logger.info(f"Model and results saved to {save_dir}")
            
            wandb.finish()

            return {
                'model': model,
                'history': history,
                'config': config,
                'data': data
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def test_model(
        model,
        data,
        config):

    print("Evaluating on test set...")

    #extract parameters
    k = config.get('k', 20)
    lambda_val = config.get('lambda_val', 1e-6)

    #get model device
    device = next(model.parameters()).device

    #move to device
    train_adj_edge_index = data['train_adj_edge_index'].to(device)
    val_adj_edge_index = data['val_adj_edge_index'].to(device)
    test_adj_edge_index = data['test_adj_edge_index'].to(device)

    # eval
    model.eval()

    with torch.no_grad():
        #evaluate on val test
        test_loss, test_ndcg = evaluation(
            model,
            data['num_users'],
            test_adj_edge_index,
            train_adj_edge_index,
            [train_adj_edge_index, val_adj_edge_index],
            k,
            lambda_val
        )

    #print results
    print(f"Test Results:")
    print(f"Loss: {test_loss:.5f}")
    print(f"NDCG@{k}: {test_ndcg:.5f}")

    #return the metrics
    return {
        'loss': test_loss,
        f'ndcg@{k}': test_ndcg
    }

def save_model(model,models_dir, data, model_config, version, history):
    """Save model, data, and training history."""
    try:

        # create the current model directory
        save_dir = os.path.join(models_dir, version)

        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), f"{save_dir}/model.pt")

        # Save data mappings and config
        with open(f"{save_dir}/lightgcn_data.pkl", 'wb') as f:
            pickle.dump({
                'user_mapping': list(data['idx_to_user_id'].values()),
                'item_mapping': list(data['idx_to_item_id'].values()),
                'config': model_config
            }, f)

        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{save_dir}/history.csv", index=False)
            
        logger.info(f"Model and results saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def train_and_savelightgcn_model(rating_repo: SQLiteRatingRepository, version: str = None, config: Dict[str, Any] = None, save_model: bool = True):
    # Load and validate config
    config = load_config(config)

    #load all config
    rating_threshold = config['rating_threshold']
    user_min_interactions = config['user_min_interactions']
    item_min_interactions = config['item_min_interactions']
    user_max_interactions = config['user_max_interactions']
    item_max_interactions = config['item_max_interactions']
    val_size = config['val_size']
    test_size = config['test_size']
    random_seed = config['random_seed']
    models_dir = config['models_dir']   
    
    # Set random seed
    set_seed(config['random_seed'])
    
    # Generate version if not provided
    if version is None:
        version = time.strftime("%Y%m%d-%H%M%S")
    elif not isinstance(version, str):
        raise ValueError("Version must be a string")
    
    logger.info(f"Starting training with version: {version}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data = prepare_lightgcn_data(
            rating_repo=rating_repo,
            rating_threshold=rating_threshold,
            user_min_interactions=user_min_interactions,
            item_min_interactions=item_min_interactions,
            user_max_interactions=user_max_interactions,
            item_max_interactions=item_max_interactions,
            val_size=val_size,
            test_size=test_size,
            random_state=random_seed
        )

        # Initialize model
        logger.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = LightGCN(
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['n_layers'],
            add_self_loops=config['add_self_loops']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # Train model
        logger.info("Training model...")
        start_time = time.time()
        history = train_lightgcn(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            data=data,
            config=config
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Evaluate model
        test_results = test_model(model, data, config)
        
        # Save model if requested
        if save_model:
            save_model(model, models_dir, data, config, version, history)
        
        return {
            'model': model,
            'history': history,
            'test_results': test_results,
            'config': config,
            'data': data,
            'version': version
        }
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
