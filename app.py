import html

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from afterglow import config, data_loader, preprocessing
from afterglow.content_model import ContentRecommender
from afterglow.mood_recommender import MoodRecommender, MOOD_PROFILES


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Afterglow | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b0d14;
        --surface: #141722;
        --border: rgba(255,255,255,0.10);
        --text: #f4f2ee;
        --muted: #a7adbd;
        --accent: #e8b86d;
        --accent2: #bd8bfa;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #22203c 0%, transparent 38%),
            radial-gradient(ellipse at 95% 20%, #29203a 0%, transparent 30%),
            var(--bg);
        color: var(--text);
    }

    h1, h2, h3, .brand, .hero-title, .movie-title {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Keep the first content safely below Streamlit's top toolbar. */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px;
        overflow: visible !important;
    }

    .brand {
        display: block;
        color: var(--accent) !important;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.20em;
        line-height: 1.8;
        margin: 0 0 0.65rem 0;
        padding-top: 0.15rem;
    }

    .brand-symbol { color: var(--accent2); }

    .hero {
        padding: 0.8rem 0 1.8rem 0;
    }

    .eyebrow {
        color: var(--accent);
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4.5rem);
        line-height: 1.08;
        font-weight: 700;
        letter-spacing: -0.045em;
        margin: 0;
        color: #f7f3ed;
    }

    .hero-title span { color: var(--accent); }

    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.8;
        max-width: 650px;
        margin-top: 1rem;
    }

    .hero-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.8;
        max-width: 650px;
        margin-top: 0.25rem;
        margin-bottom: 1.8rem;
    }

    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        line-height: 1.4;
        margin-top: 0.65rem;
        color: var(--text);
        overflow-wrap: anywhere;
    }

    .genre-pill {
        display: inline-block;
        background: #27263a;
        border: 1px solid rgba(189,139,250,0.22);
        color: #d5c4fa;
        border-radius: 999px;
        padding: 4px 9px;
        margin: 2px 4px 3px 0;
        font-size: 0.70rem;
    }

    .score {
        display: inline-block;
        border: 1px solid rgba(232,184,109,0.3);
        background: rgba(232,184,109,0.10);
        color: #f1ca8b;
        padding: 5px 9px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .muted {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.7;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(20,23,34,0.86);
        border-color: var(--border);
        border-radius: 16px;
    }

    [data-testid="stSidebar"] {
        background: rgba(17,19,29,0.98);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text);
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.5rem;
    }

    div.stButton > button[kind="primary"] {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #17130d;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        padding: 15px;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] { color: var(--accent); }

    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .hero-title { font-size: 2.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_movie_data():
    """Fetch and preprocess popular movies from TMDB."""
    movies = data_loader.fetch_popular_movies(pages=3)
    if not movies:
        return pd.DataFrame()

    df = pd.DataFrame(movies)
    if "id" not in df.columns:
        raise ValueError("TMDB response is missing movie IDs.")

    df = df.rename(columns={"id": "movieId"})
    df = preprocessing.clean_movies(df)
    return df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)


# =========================================================
# HELPERS
# =========================================================

def safe_text(value, fallback=""):
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    return value if value else fallback


def get_genres(row):
    genres = row.get("genre_list", [])
    if isinstance(genres, list):
        return [str(g).strip() for g in genres if isinstance(g, str) and g.strip()]
    if isinstance(genres, str):
        return [g.strip() for g in genres.split("|") if g.strip()]
    return []


def get_year(row):
    year = row.get("year")
    if pd.notna(year):
        try:
            return str(int(year))
        except (ValueError, TypeError, OverflowError):
            pass
    return "Year unavailable"


def get_poster_url(poster_path):
    if not isinstance(poster_path, str) or not poster_path.strip():
        return None
    if poster_path.startswith(("https://", "http://")):
        return poster_path
    return f"{config.TMDB_IMAGE_BASE_URL.rstrip('/')}/{poster_path.lstrip('/')}"


def normalise_series(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (values - minimum) / (maximum - minimum)


def make_quality_scores(movies_df):
    """Build a seed-independent baseline from popularity and rating."""
    popularity = normalise_series(
        movies_df.get("popularity", pd.Series(0, index=movies_df.index))
    )
    ratings = normalise_series(
        movies_df.get("vote_average", pd.Series(0, index=movies_df.index))
    )
    vote_counts = pd.to_numeric(
        movies_df.get("vote_count", pd.Series(0, index=movies_df.index)),
        errors="coerce",
    ).fillna(0)

    confidence = (vote_counts / (vote_counts + 20)).clip(0, 1)
    adjusted_rating = ratings * confidence + 0.5 * (1 - confidence)
    quality = 0.55 * popularity + 0.45 * adjusted_rating

    return {
        movie_id: float(score)
        for movie_id, score in zip(movies_df["movieId"], quality)
    }


def make_seed_scores(content_model, seed_id, movies_df):
    """Calculate similarity between a chosen movie and the catalogue."""
    if seed_id not in content_model.id_to_index:
        raise KeyError(f"Starting movie ID {seed_id} was not found.")

    seed_index = content_model.id_to_index[seed_id]
    similarities = cosine_similarity(
        content_model.tfidf_matrix[seed_index],
        content_model.tfidf_matrix,
    ).ravel()

    return {
        movie_id: float(similarities[index])
        for index, movie_id in enumerate(movies_df["movieId"])
    }


def make_mood_content_scores(content_model, movies_df, mood, quality_scores):
    """
    Build a seed-free content baseline using mood-related genres where
    the profile format supports it. If the profile format differs,
    fall back safely to the quality baseline.
    """
    profile = MOOD_PROFILES.get(mood, {})
    genre_weights = {}

    if isinstance(profile, dict):
        possible_genres = profile.get("genres", {})
        if isinstance(possible_genres, dict):
            genre_weights = possible_genres

    query_terms = []
    for genre, weight in genre_weights.items():
        try:
            repetitions = max(1, min(5, round(float(weight) * 3)))
        except (TypeError, ValueError):
            repetitions = 1
        query_terms.extend([str(genre)] * repetitions)

    vectorizer = getattr(content_model, "vectorizer", None)
    matrix = getattr(content_model, "tfidf_matrix", None)

    if not query_terms or vectorizer is None or matrix is None:
        return quality_scores.copy()

    try:
        query_vector = vectorizer.transform([" ".join(query_terms)])
        similarities = cosine_similarity(query_vector, matrix).ravel()
        return {
            movie_id: float(
                0.75 * similarities[index]
                + 0.25 * quality_scores.get(movie_id, 0.5)
            )
            for index, movie_id in enumerate(movies_df["movieId"])
        }
    except (AttributeError, ValueError, TypeError):
        return quality_scores.copy()


# =========================================================
# MOVIE CARD
# =========================================================

def render_movie_card(row, watched_ids, score_label):
    movie_id = row["movieId"]
    title = safe_text(row.get("title"), "Untitled movie")
    overview = safe_text(
        row.get("overview"),
        "No description is available for this movie yet.",
    )
    poster_url = get_poster_url(row.get("poster_path"))
    genres = get_genres(row)
    year = get_year(row)

    with st.container(border=True):
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="height:250px;display:flex;align-items:center;
                justify-content:center;background:#202331;border-radius:10px;
                color:#a7adbd;">🎬 Poster unavailable</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="movie-title">{html.escape(title)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(year)

        if genres:
            badges = "".join(
                f'<span class="genre-pill">{html.escape(genre)}</span>'
                for genre in genres[:3]
            )
            st.markdown(badges, unsafe_allow_html=True)

        try:
            rating = float(row.get("vote_average"))
            if pd.notna(rating) and rating > 0:
                st.markdown(
                    f'<span class="score">★ {rating:.1f} / 10</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Rating unavailable")
        except (ValueError, TypeError):
            st.caption("Rating unavailable")

        st.write(overview if len(overview) <=
                 180 else overview[:177].rstrip() + "...")

        final_score = row.get("final_score")
        similarity_score = row.get("similarity_score")

        if final_score is not None and pd.notna(final_score):
            st.caption(f"{score_label}: {float(final_score):.3f}")
        elif similarity_score is not None and pd.notna(similarity_score):
            st.caption(f"Content similarity: {float(similarity_score):.3f}")

        if movie_id in watched_ids:
            st.button(
                "✓ Watched",
                key=f"watched_{movie_id}",
                disabled=True,
                use_container_width=True,
            )
        elif st.button(
            "Mark as watched",
            key=f"watch_{movie_id}",
            use_container_width=True,
        ):
            st.session_state.watched_ids.append(movie_id)
            st.rerun()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():
    if "watched_ids" not in st.session_state:
        st.session_state.watched_ids = []

    # Use native Streamlit text components for the hero heading so HTML
    # tags cannot accidentally appear as visible text in the browser.
    st.markdown(
        '<div class="brand">AFTERGLOW <span class="brand-symbol">✦</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eyebrow">YOUR PERSONAL CINEMA COMPANION</div>',
        unsafe_allow_html=True,
    )
    st.title("Films for your current feeling.")
    st.markdown(
        """
        <div class="hero-description">
            Every mood has a movie. Discover your next favourite through
            mood-aware recommendations, content similarity, and movie
            quality signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # SIDEBAR CONTROLS
    # -----------------------------------------------------
    with st.sidebar:
        st.markdown("## Your experience")
        st.caption("Tune your discovery session.")

        if config.validate_config():
            st.success("TMDB credentials configured")
        else:
            st.warning("TMDB credentials missing")

        st.divider()
        st.markdown("### 🎭 Your mood")

        mood_options = ["No mood preference", *list(MOOD_PROFILES.keys())]
        selected_mood = st.selectbox(
            "What would you like to watch?",
            options=mood_options,
            index=1 if len(mood_options) > 1 else 0,
            key="afterglow_mood",
        )

        st.markdown("### 🧠 Recommendation engine")
        with st.expander("Advanced recommendation settings", expanded=False):
            strategy = st.radio(
                "Choose a recommendation method",
                options=[
                    "Mood Discovery",
                    "Hybrid · Mood + Content",
                    "Content Similarity",
                ],
                index=0,
                key="afterglow_strategy",
            )
            top_k = st.slider(
                "Movies to show",
                min_value=3,
                max_value=18,
                value=6,
                key="afterglow_top_k",
            )

        # Defaults used on the initial run before the expander is opened.
        strategy = st.session_state.get("afterglow_strategy", "Mood Discovery")
        top_k = st.session_state.get("afterglow_top_k", 6)

        st.divider()
        st.markdown("### 👀 Your watch history")
        st.metric("Movies watched", len(st.session_state.watched_ids))

        if st.session_state.watched_ids and st.button(
            "Clear watched history",
            use_container_width=True,
        ):
            st.session_state.watched_ids = []
            st.rerun()

        st.caption("Watch history lasts for the current Streamlit session.")

        with st.expander("Current settings"):
            st.write(f"**Mood:** {selected_mood}")
            st.write(f"**Engine:** {strategy}")
            st.write(f"**Movies:** {top_k}")

    # -----------------------------------------------------
    # LOAD MOVIES
    # -----------------------------------------------------
    with st.spinner("Finding films for your next movie night..."):
        try:
            movies_df = load_movie_data()
        except Exception as err:
            st.error(
                "We couldn't load movies from TMDB. "
                "Check your connection and API configuration."
            )
            with st.expander("Technical details"):
                st.code(str(err))
            if st.button("Retry loading movies"):
                load_movie_data.clear()
                st.rerun()
            return

    if movies_df.empty:
        st.warning("No movies were returned. Please try again later.")
        return

    if "movieId" not in movies_df.columns:
        st.error("Movie data is missing the movieId column.")
        return

    # -----------------------------------------------------
    # INITIALISE MODELS
    # -----------------------------------------------------
    try:
        content_model = ContentRecommender()
        content_model.fit(movies_df, id_col="movieId")
        mood_model = MoodRecommender()
    except Exception as err:
        st.error("The recommendation models could not be initialized.")
        with st.expander("Technical details"):
            st.code(str(err))
        return

    # -----------------------------------------------------
    # STARTING MOVIE: ONLY FOR CONTENT SIMILARITY
    # -----------------------------------------------------
    seed_id = None
    title_by_id = {}

    if strategy == "Content Similarity":
        st.markdown("### 🎞️ Choose a film you enjoy")
        st.caption("Recommendations will be based on your selected film.")

        title_by_id = {
            row["movieId"]: (
                f"{safe_text(row.get('title'), 'Untitled movie')} "
                f"({get_year(row)})"
            )
            for _, row in movies_df.iterrows()
        }
        seed_ids = list(title_by_id.keys())

        if not seed_ids:
            st.warning("No starting movies are available.")
            return

        seed_id = st.selectbox(
            "Use this movie as your starting point",
            options=seed_ids,
            index=0,
            format_func=lambda movie_id: title_by_id[movie_id],
            key="afterglow_seed_movie",
        )

    # -----------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # -----------------------------------------------------
    active_mood = None if selected_mood == "No mood preference" else selected_mood
    watched_ids = list(st.session_state.watched_ids)

    try:
        quality_scores = make_quality_scores(movies_df)

        if strategy == "Mood Discovery":
            recommended_df = mood_model.recommend(
                candidate_movies=movies_df,
                mood=active_mood,
                watched_ids=watched_ids,
                base_scores=quality_scores,
                id_col="movieId",
                top_n=top_k,
                mood_weight=0.80,
            )
            score_label = "Mood discovery score"

        elif strategy == "Hybrid · Mood + Content":
            if active_mood is None:
                hybrid_scores = quality_scores.copy()
                mood_weight = 0.0
            else:
                hybrid_scores = make_mood_content_scores(
                    content_model,
                    movies_df,
                    active_mood,
                    quality_scores,
                )
                mood_weight = 0.55

            recommended_df = mood_model.recommend(
                candidate_movies=movies_df,
                mood=active_mood,
                watched_ids=watched_ids,
                base_scores=hybrid_scores,
                id_col="movieId",
                top_n=top_k,
                mood_weight=mood_weight,
            )
            score_label = "Hybrid recommendation score"

        else:
            recommended_df = content_model.recommend_similar(
                movie_id=seed_id,
                top_n=top_k,
                exclude_ids=watched_ids,
            )
            score_label = "Content similarity score"

    except Exception as err:
        st.error("We couldn't generate recommendations.")
        with st.expander("Technical details"):
            st.code(str(err))
        return

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------
    st.divider()
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Movies available", len(movies_df))
    metric2.metric("Recommendations", len(recommended_df))
    metric3.metric("Movies watched", len(st.session_state.watched_ids))

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
    st.markdown("## ✨ Your recommendations")

    if strategy == "Mood Discovery":
        if active_mood:
            st.caption(
                f"ACTIVE ENGINE: MOOD DISCOVERY · MOOD: {active_mood}. "
                "Films are ranked by mood alignment and movie quality."
            )
        else:
            st.caption(
                "ACTIVE ENGINE: MOOD DISCOVERY · No mood selected. "
                "Films are ranked using popularity and rating signals."
            )
    elif strategy == "Hybrid · Mood + Content":
        if active_mood:
            st.caption(
                f"ACTIVE ENGINE: HYBRID · MOOD: {active_mood}. "
                "Ranking combines mood alignment, mood-based content "
                "relevance, and movie quality."
            )
        else:
            st.caption(
                "ACTIVE ENGINE: HYBRID · No mood selected. "
                "Ranking uses movie quality signals."
            )
    else:
        st.caption(
            "ACTIVE ENGINE: CONTENT SIMILARITY · Finding films similar to "
            f"{title_by_id.get(seed_id, 'your selected film')}."
        )

    if recommended_df.empty:
        st.info(
            "No recommendations are available with these settings. "
            "Try another mood or clear your watch history."
        )
        return

    with st.expander("View recommendation scores"):
        score_columns = [
            column for column in [
                "title",
                "similarity_score",
                "base_score",
                "mood_score",
                "final_score",
            ]
            if column in recommended_df.columns
        ]
        if score_columns:
            st.dataframe(
                recommended_df[score_columns],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "This recommendation method does not expose a score table.")

    cols = st.columns(3)
    for index, (_, movie) in enumerate(recommended_df.iterrows()):
        with cols[index % len(cols)]:
            render_movie_card(
                movie,
                st.session_state.watched_ids,
                score_label,
            )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------
    st.divider()
    st.markdown(
        """
        <div class="muted" style="text-align:center;padding:1rem 0;">
            AFTERGLOW ✦ Made for movie lovers<br>
            Powered by TMDB and machine learning
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
