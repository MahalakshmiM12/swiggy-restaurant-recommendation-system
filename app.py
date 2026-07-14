"""
Swiggy's Restaurant Recommendation System — Streamlit App
==========================================================
Loads the artifacts produced by swiggy_restaurant_recommendation_system.ipynb
(cleaned_data.csv, encoded_data.csv, encoder.pkl, kmeans_model.pkl) and lets a
user get restaurant recommendations based on city, cuisine, rating and budget
preferences, using the same cosine-similarity content-based engine built in
the notebook.
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Data / artifact loading (cached so the app stays fast after first load)
# ----------------------------------------------------------------------
@st.cache_data
def load_cleaned_data():
    return pd.read_csv("cleaned_data.csv", index_col=0)


@st.cache_data
def load_encoded_data():
    return pd.read_csv("encoded_data.csv", index_col=0)


@st.cache_resource
def load_encoder():
    with open("encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_kmeans():
    with open("kmeans_model.pkl", "rb") as f:
        return pickle.load(f)


df = load_cleaned_data()
encoded_df = load_encoded_data()
encoder_bundle = load_encoder()
kmeans = load_kmeans()

city_ohe = encoder_bundle["city_ohe"]
mlb = encoder_bundle["mlb"]
scaler = encoder_bundle["scaler"]
top_cities = encoder_bundle["top_cities"]
num_cols = encoder_bundle["num_cols"]


# ----------------------------------------------------------------------
# Recommendation engine (mirrors the notebook exactly, for consistency)
# ----------------------------------------------------------------------
def recommend_restaurants(city=None, cuisines=None, min_rating=0.0, max_cost=None, top_n=10):
    city_bucket = city if city in top_cities else "Other"
    q_city_vec = city_ohe.transform(pd.DataFrame([{"city_bucket": city_bucket}]))
    q_cuisine_vec = mlb.transform([cuisines if cuisines else []])

    target_rating = min_rating if min_rating else df["rating"].median()
    target_cost = max_cost if max_cost else df["cost"].median()
    target_count = df["rating_count_num"].median()
    q_num_vec = scaler.transform(
        pd.DataFrame([[target_rating, target_count, target_cost]], columns=num_cols)
    )

    q_vec = np.hstack([q_num_vec, q_city_vec, q_cuisine_vec]).astype(np.float32)

    mask = pd.Series(True, index=df.index)
    if city:
        mask &= df["city"] == city
    if max_cost:
        mask &= df["cost"] <= max_cost
    if min_rating:
        mask &= df["rating"] >= min_rating

    widened = False
    if mask.sum() < top_n:
        mask = pd.Series(True, index=df.index)
        widened = True

    cand_idx = df.index[mask]
    sims = cosine_similarity(q_vec, encoded_df.loc[cand_idx].values).flatten()
    order = np.argsort(-sims)[:top_n]
    result_idx = cand_idx[order]

    result = df.loc[
        result_idx,
        ["name", "city", "rating", "rating_count", "cost", "cuisine", "address", "link", "cluster"],
    ].copy()
    result["similarity"] = sims[order].round(3)
    return result.reset_index(drop=True), widened


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🍽️ Swiggy Restaurant Recommendation System")
st.caption(
    "Content-based recommendations using K-Means market segmentation + cosine similarity, "
    f"over {len(df):,} cleaned Swiggy restaurant listings."
)

tab_recommend, tab_insights, tab_about = st.tabs(["🔎 Get Recommendations", "📊 Market Insights", "ℹ️ About"])

# ---- Tab 1: Recommendations ----
with tab_recommend:
    st.subheader("Tell us what you're in the mood for")

    col1, col2 = st.columns(2)

    with col1:
        city_options = ["Any city"] + sorted(df["city"].unique().tolist())
        city_choice = st.selectbox(
            "City / locality",
            options=city_options,
            index=0,
            help="Data is at locality granularity for big metros, e.g. 'Koramangala,Bangalore'.",
        )
        selected_city = None if city_choice == "Any city" else city_choice

        all_cuisines = sorted({c for tags in df["cuisine"].str.split(",") for c in tags})
        cuisine_choice = st.multiselect(
            "Cuisine preference(s)",
            options=all_cuisines,
            default=[],
        )

    with col2:
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 4.0, 0.1)
        max_cost = st.slider("Maximum cost for two (₹)", 50, 2000, 500, 50)
        top_n = st.slider("Number of recommendations", 5, 25, 10)

    if st.button("Get Recommendations", type="primary"):
        results, widened = recommend_restaurants(
            city=selected_city,
            cuisines=cuisine_choice,
            min_rating=min_rating,
            max_cost=max_cost,
            top_n=top_n,
        )

        if widened:
            st.info(
                "Your filters were narrower than the number of results requested, "
                "so the search was widened beyond your exact constraints to still "
                "show the closest matches."
            )

        st.subheader(f"Top {len(results)} recommendations")
        for _, row in results.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{row['name']}** — {row['city']}")
                    st.markdown(f"🍴 {row['cuisine']}  |  💰 ₹{row['cost']:.0f} for two  |  ⭐ {row['rating']} ({row['rating_count']})")
                    st.caption(row["address"])
                    st.markdown(f"[View on Swiggy]({row['link']})")
                with c2:
                    st.metric("Match score", f"{row['similarity']:.0%}")
                    st.caption(f"Segment #{row['cluster']}")
    else:
        st.info("Set your preferences above and click **Get Recommendations**.")

# ---- Tab 2: Market insights (from clustering) ----
with tab_insights:
    st.subheader("Restaurant market segments (K-Means clusters)")

    profile = df.groupby("cluster").agg(
        n_restaurants=("id", "count"),
        avg_rating=("rating", "mean"),
        avg_cost=("cost", "mean"),
    ).round(2)
    profile["top_cuisine"] = df.groupby("cluster")["cuisine"].agg(
        lambda s: s.str.split(",").explode().mode().iloc[0]
    )
    st.dataframe(profile, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Restaurants per cluster**")
        st.bar_chart(profile["n_restaurants"])
    with col2:
        st.markdown("**Average cost per cluster (₹)**")
        st.bar_chart(profile["avg_cost"])

    st.markdown("**Top cuisines overall**")
    top_cuisines = df["cuisine"].str.split(",").explode().str.strip().value_counts().head(15)
    st.bar_chart(top_cuisines)

# ---- Tab 3: About ----
with tab_about:
    st.markdown(
        """
        ### About this project

        This app is the Streamlit front-end for the **Swiggy Restaurant Recommendation System**
        capstone. The full pipeline (data cleaning, EDA, encoding, clustering, and the
        recommendation engine) lives in
        `swiggy_restaurant_recommendation_system.ipynb`.

        **Pipeline summary:**
        - Cleaned {n_rows:,} restaurant listings (duplicates dropped, missing values
          reasonably imputed city-by-city, promotional text filtered out of the cuisine field).
        - Encoded `city` (top-150 + "Other") and `cuisine` (multi-label) with One-Hot /
          Multi-Hot encoding; standardized `rating`, `rating_count`, `cost`.
        - Segmented the market into **{k} clusters** with MiniBatchKMeans.
        - Recommendations are ranked by cosine similarity between your stated preferences and
          each restaurant's encoded profile, then mapped back to the human-readable cleaned
          dataset.

        **Artifacts used by this app:** `cleaned_data.csv`, `encoded_data.csv`, `encoder.pkl`,
        `kmeans_model.pkl`.
        """.format(n_rows=len(df), k=df["cluster"].nunique())
    )
