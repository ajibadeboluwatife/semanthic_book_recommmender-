#install relevant packages
import streamlit as st
import pandas as pd
from src.search import SemanticSearcher

st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

@st.cache_resource
def get_searcher():
    return SemanticSearcher("data")

st.title("📚 Semantic Book Recommender")

st.write("Type what you're in the mood for (themes, vibes, topics, plots). We'll match semantically, not just by keywords.")

query = st.text_input("Your query", value="space survival with humor", placeholder="e.g., 'cozy fantasy about found family', 'database systems at scale'")

col1, col2, col3 = st.columns(3)
with col1:
    # Options are populated dynamically after the first run; leaving empty here
    selected_genres = st.multiselect("Filter: Genres", options=[])
with col2:
    year_min, year_max = st.slider("Filter: Publication Year Range", 1950, 2025, (1990, 2025))
with col3:
    min_rating = st.slider("Filter: Min Rating", 0.0, 5.0, 0.0, 0.1)

alpha = st.slider("Hybrid weighting (Semantic ↔ Keyword)", 0.0, 1.0, 0.7, 0.05)

if st.button("Search") and query.strip():
    searcher = get_searcher()
    # Populate genre options dynamically after loading
    all_genres = sorted(set(
        g.strip() for s in searcher.df["genres"].fillna("") for g in s.split(",")
    ) - set([""]))
    genres = st.multiselect("Filter: Genres", options=all_genres, default=selected_genres)

    filters = {
        "genres": genres,
        "year_min": year_min,
        "year_max": year_max,
        "min_rating": min_rating,
    }
    results = searcher.query(query, top_k=12, filters=filters, hybrid_alpha=alpha)
    if results.empty:
        st.warning("No matches. Try removing filters or changing your query.")
    else:
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"### {row['title']}  \\n"
                            f"**Author:** {row['author']}  \\n"
                            f"**Year:** {int(row['year'])} | **Rating:** {row['rating']}  \\n"
                            f"**Genres:** {row['genres']}  \\n"
                            f"**ISBN-13:** {row['isbn13']}  \\n"
                            f"**Why this match (score):** {row['score']:.3f}")
                st.write(row['description'])
                st.divider()
else:
    st.info("Enter a query and click **Search** to see recommendations.")
