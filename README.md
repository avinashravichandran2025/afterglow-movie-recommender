# Afterglow ✨

## Mood-Aware Movie Recommendation System

Afterglow is a machine learning movie recommendation application that helps users discover films based on their current mood and movie preferences.

The application combines movie metadata from The Movie Database (TMDB) with content-based recommendation techniques to deliver personalised movie suggestions through an interactive Streamlit interface.

🚀 **Live Demo:** [Try Afterglow](https://afterglow-movie-recommender-eu5xfjndpjiqfr8uefd5rx.streamlit.app/)

💻 **GitHub Repository:** [View Source Code](https://github.com/avinashravichandran2025/afterglow-movie-recommender)

---

## ✨ Features

- **Live Movie Data:** Fetch popular movies, posters, ratings, descriptions, and metadata from TMDB.
- **Content-Based Filtering:** Identify movies with similar genres and descriptions using content similarity techniques.
- **Mood-Aware Recommendations:** Discover movies based on moods such as Uplifting, Dark, Mind-Bending, Relaxed, and Romantic.
- **Recommendation Ranking:** Combine mood alignment, content relevance, and movie quality signals.
- **Movie Similarity Search:** Discover similar films using the available recommendation features.
- **Watch History:** Track watched movies during the current Streamlit session.
- **Interactive Interface:** Explore recommendations through a dark-themed Streamlit web application.
- **Testing:** Run automated tests for supported components.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Recommendation Techniques | Content Similarity, Mood-Based Ranking |
| Movie Data | TMDB API |
| Testing | Pytest |
| Version Control | Git, GitHub |

## 📸 Application Preview

<p align="center">
  <a href="https://afterglow-movie-recommender-eu5xfjndpjiqfr8uefd5rx.streamlit.app/">
    <img src="https://github.com/avinashravichandran2025/afterglow-movie-recommender/blob/main/assets/afterglow-home.png?raw=true" alt="Afterglow Movie Recommendation Application" width="800">
  </a>
</p>

## 📁 Project Structure

```text
afterglow-movie-recommender/
├── assets/
├── src/
│   └── afterglow/
│       ├── __init__.py
│       ├── collaborative_model.py
│       ├── config.py
│       ├── content_model.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── experiments.py
│       ├── mood_recommender.py
│       └── preprocessing.py
├── tests/
├── app.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

_Note: The structure above describes the intended project organisation. Keep only files and directories that actually exist in your repository._

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later, compatible with the dependencies in `requirements.txt`
- A TMDB API credential
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/avinashravichandran2025/afterglow-movie-recommender.git
cd afterglow-movie-recommender
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure TMDB Credentials

Create a `.env` file in the project root directory:

```env
TMDB_READ_TOKEN=your_tmdb_read_access_token
```

Use the environment variable names expected by `src/afterglow/config.py`.

**Security:** Never commit your `.env` file, API tokens, or other credentials to GitHub.

### 5. Run the Application

```bash
PYTHONPATH=src python -m streamlit run app.py
```

Open the local URL displayed in your terminal to use Afterglow.

## 🧠 How It Works

### Content-Based Filtering

The content recommendation model transforms movie genres and descriptions into TF-IDF feature vectors. Cosine similarity measures how closely two movies match, allowing users to discover films with similar content.

### Mood-Aware Ranking

The mood recommendation engine adjusts movie rankings according to the selected mood. Genre alignment and movie quality signals help prioritise relevant recommendations.

### Hybrid Recommendations

The hybrid approach combines mood alignment with content relevance and quality signals to provide another way to discover films.

### Watched History

Users can mark recommendations as watched. These movies are excluded from future recommendations during the current application session.

## 🧪 Running Tests

Run the test suite from the project root:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Check that the tests pass before reporting the results in this README.

## 🔐 Environment Variables

Store API credentials locally in `.env`.

Example `.env.example`:

```env
TMDB_READ_TOKEN=replace_with_your_tmdb_read_access_token
```

The example file must contain placeholders only, never real credentials.

## 👨‍💻 Author

**Avinash Ravichandran**

GitHub: [@avinashravichandran2025](https://github.com/avinashravichandran2025)

## 📜 License

📜 License

This project is available for viewing and educational purposes. No open-source license has been added yet. All rights are reserved by the author unless otherwise stated.
