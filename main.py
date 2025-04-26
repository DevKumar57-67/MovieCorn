# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Load & Prepare the Data --------------
@st.cache_data
def load_data():
    movies = pd.read_csv("netflix_titles.csv")
    movies = movies.drop(['show_id', 'director', 'cast', 'country', 'date_added', 'rating',
                          'release_year', 'duration', 'listed_in'], axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['description'])
    cos_theta = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return movies, cos_theta

movies, cos_theta = load_data()

# ----------- Recommendation Function --------------
def recommend_movies_by_description(movie_title, num_recommendations=5):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cos_theta[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except IndexError:
        return None

# ----------- Streamlit UI --------------
# Page config
st.set_page_config(page_title="Netflix Movie Recommender 🍿", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for style
st.markdown("""
    <style>
    .main {
        background-color: #141414;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #E50914;
    }
    .stButton>button {
        color: white;
        background-color: #E50914;
        border: None;
        border-radius: 8px;
        padding: 0.75em 1em;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🎬 Netflix Movie Recommender")

st.markdown("### Select a movie you liked recently:")
selected_movie = st.selectbox("Movie Title", sorted(movies['title'].dropna().unique()))

# Number of recommendations
num_recommendations = st.slider("How many recommendations would you like?", 1, 10, 5)

# Button to trigger recommendation
if st.button("Recommend 🎯"):
    with st.spinner("Finding the best matches for you... 🎥"):
        recommendations = recommend_movies_by_description(selected_movie, num_recommendations)

    if recommendations:
        st.success("Here are your recommended movies:")
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"**{i}. {movie}**")
    else:
        st.error("Sorry, couldn't find the movie in our database. Try another one!")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by CheekyDev</center>", unsafe_allow_html=True)
