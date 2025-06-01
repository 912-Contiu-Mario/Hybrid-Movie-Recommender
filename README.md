# Hybrid Movie Recommender System

This project implements a hybrid movie recommender that combines **LightGCN collaborative filtering** with **MPNet-based semantic embeddings**. This approach enhances recommendation quality (especially in sparse data scenarios) and significantly expands the catalogue of recommendable movies.

## Architecture Overview
- **Backend**: FastAPI-based service leveraging Faiss for efficient similarity search and PostgreSQL for data persistence.  
  *Located in `/backend`*
- **Frontend**: Responsive React + TypeScript UI with TMDB API metadata integration.  
  *Located in `/frontend`*
- **Notebooks**: Jupyter notebooks for LightGCN training, dataset filtering, and MPNet embedding generation.  
  *Located in `/notebooks`*

## How It Works
The system dynamically combines two recommendation strategies:

1. **LightGCN**  
   Graph-based collaborative filtering trained on user-item interactions.  
   ✓ Excels at finding behavioral patterns  
   ✗ Limited to ~4,500 training movies

2. **MPNet Semantic Embeddings**  
   Content-based vectors from movie metadata.  
   ✓ Recommends from ~45,000 movies  
   ✓ Addresses cold-start for new/less-rated movies  

**Hybrid Approach**: Blends both strategies, achieving **up to 30% improvement** in key metrics under sparse data conditions.

## Key Benefits
✅ **Expanded Catalogue**: 10x more movies (45K vs 4.5K)  
✅ **Sparse Data Performance**: 30% metric improvement with limited interactions  
✅ **Cold-Start Resilience**: Better recommendations for new movies  
✅ **Effective Signal Blending**: Combines behavioral + content-based signals  

---

## Backend Details
**Core Stack**: FastAPI, Faiss, PostgreSQL  
**Key Features**:
- **Pseudo Real-Time Recommendations**:  
  Re-runs inference every 60s using latest user ratings
- **Concurrency Management**:  
  Read-write locks for safe model access during updates
- **Zero-Downtime Model Reload**:  
  Hot-swap updated models via API endpoint  
**Future Enhancement**: Periodic full model retraining

---

## Frontend Highlights
**Tech Stack**: React + TypeScript  
**Features**:
- **Rich Metadata**: TMDB API integration for posters/descriptions
- **Responsive Design**: Works on mobile/tablet/desktop
- **Core Functionalities**:
  - 🔐 User registration/login
  - ⭐ Movie rating system
  - 📊 Rated movie history
  - 💡 Recommendation sources:
    - Hybrid (LightGCN + MPNet)
    - LightGCN-only
    - Content-based (MPNet)
  - 🎬 "Find similar movies" for any film
 
![hybrid_movie_rec_frontpage](https://github.com/user-attachments/assets/5e633a7d-e258-4dec-b517-fe85b9c838d6)

![hybrid_movie_rec_moviepage](https://github.com/user-attachments/assets/65a84ffc-da6b-4936-a3f5-0aa47d491a37)


---

## Project Goal
Demonstrate how **combining graph-based collaborative filtering** with **semantic understanding** creates a more flexible, scalable, and effective recommendation system—especially for overcoming data sparsity and cold-start challenges.

> Explore the codebase and notebooks in their respective directories for implementation details.
