# Afterglow ✨
### Mood-Aware Movie Recommendation System

Afterglow is a machine learning movie recommendation application that helps users discover films based on their current mood and movie preferences.

The application combines live movie metadata from The Movie Database (TMDB) with content-based recommendation techniques to deliver personalised movie suggestions through an interactive Streamlit interface.

## ✨ Features

- **Live Movie Data:** Fetch popular movies, posters, ratings, descriptions, and metadata from TMDB.
- **Content-Based Filtering:** Use TF-IDF vectorisation and cosine similarity to identify movies with similar genres and descriptions.
- **Mood-Aware Recommendations:** Discover movies based on moods such as Uplifting, Dark, Mind-Bending, Relaxed, and Romantic.
- **Hybrid Recommendation Engine:** Combine mood alignment, content relevance, and movie quality signals.
- **Movie Similarity Search:** Select a movie and discover similar films.
- **Watched History:** Mark movies as watched and exclude them from subsequent recommendations during the current session.
- **Interactive Interface:** Explore recommendations through a dark-themed Streamlit web application.
- **Testing:** Automated tests for supported data-processing, recommendation, and evaluation components.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Recommendation Algorithm | TF-IDF, Cosine Similarity, Mood-Based Ranking |
| Movie Data | TMDB API |
| Testing | Pytest |
| Version Control | Git, GitHub |

## 📸 Application Preview

<!-- Add a screenshot of your running application here. -->
<!-- Save the screenshot in the repository as assets/afterglow-preview.png. -->

![Afterglow application preview](assets/afterglow-preview.png)

## 📁 Project Structure

```text
afterglow-movie-recommender/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── src/
│   └── afterglow/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── content_model.py
│       ├── collaborative_model.py
│       ├── mood_recommender.py
│       ├── evaluation.py
│       └── experiments.py
├── tests/
└── data/
```

*Note: The structure above describes the intended project organisation. Keep only files and directories that actually exist in your repository.*

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

Add a license file if you intend to distribute this project under an open-source license.
