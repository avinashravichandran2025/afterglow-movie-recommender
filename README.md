# Afterglow ✨ — Machine Learning Movie Recommendation System

**Afterglow** is an end-to-end Machine Learning movie recommendation system designed for portfolio and production environments. It blends live movie metadata from **The Movie Database (TMDB)** with historical user interaction patterns from **MovieLens** using Content-Based Filtering (TF-IDF + Cosine Similarity), Collaborative Filtering (SVD Matrix Factorization), and a custom Mood-Aware Ranking Engine.

---

## 🌟 Key Features

- **Live TMDB Integration**: Real-time fetching of popular movie metadata, high-res posters, and genre details with automatic fallback HTTP engines for SSL resilience across environments.
- **Content-Based Filtering**: TF-IDF vectorization over movie genres and synopsis descriptions paired with Cosine Similarity.
- **Collaborative Filtering**: Singular Value Decomposition (SVD) matrix factorization on historical rating interaction matrices.
- **Mood-Aware Ranking Engine**: Dynamic mood profiles (*Uplifting*, *Dark/Gritty*, *Mind-Bending Sci-Fi*, *Relaxed*, *Romantic*) that re-weight candidate scores dynamically.
- **Watched History Controls**: Instant exclusion of already watched movies from recommendations.
- **Full Test Suite & Metrics**: Automated unit testing (`pytest`) verifying evaluation metrics (`Precision@K`, `Recall@K`, `NDCG@K`), preprocessing pipelines, and model outputs.
- **Interactive Streamlit App**: Modern dark-mode UI with glassmorphism aesthetics.

---

## 📁 Repository Structure

```
afterglow/
├── data/
│   ├── raw/          # MovieLens raw datasets (git-ignored)
│   └── processed/    # Cleaned datasets and cached metadata
├── src/afterglow/
│   ├── __init__.py
│   ├── config.py             # Credential manager & environment paths
│   ├── data_loader.py        # TMDB API client & MovieLens loader
│   ├── preprocessing.py      # Genre parsing, title/year extraction & cleaning
│   ├── content_model.py      # TF-IDF + Cosine Similarity Recommender
│   ├── collaborative_model.py # SVD Matrix Factorization Recommender
│   ├── mood_recommender.py   # Hybrid mood weighting & watched filtering
│   ├── evaluation.py         # Precision@K, Recall@K & NDCG@K metrics
│   └── experiments.py        # Benchmark experiment runner
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_content_model.py
│   ├── test_collaborative_model.py
│   ├── test_recommendations.py
│   └── test_evaluation.py
├── app.py              # Streamlit Web Application UI
├── requirements.txt    # Project dependencies
├── .env                # API Keys (TMDB_READ_TOKEN / TMDB_API_KEY)
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/your-username/afterglow.git
cd afterglow

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure TMDB API Key
Add your TMDB API Read Token or API Key in `.env`:
```env
TMDB_READ_TOKEN=your_bearer_token_here
```

### 3. Run Unit Tests
```bash
PYTHONPATH=src pytest tests/
```

### 4. Launch the Interactive App
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

The system measures recommendation accuracy using top-K ranking metrics:
- **Precision@K**: Fraction of top-$K$ recommended movies that the user rated positively ($\ge 3.5$).
- **Recall@K**: Fraction of user's total liked movies captured within the top-$K$ recommendations.
- **NDCG@K**: Normalized Discounted Cumulative Gain accounting for position-dependent ranking quality.

---

## 📜 License
MIT License © 2026 Afterglow Team
