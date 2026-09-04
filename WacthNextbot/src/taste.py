import re

def clean_title(title: str) -> str:
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip() 

def build_soup(features: dict) -> str: 
    def clean(text: str) -> str:
        return text.lower().replace(" ", "")
    
    genres   = [clean(g) for g in features.get("genres", [])]
    keywords = [clean(k) for k in features.get("keywords", [])]
    cast     = [clean(c) for c in features.get("cast", [])]
    director = clean(features.get("director", ""))

    tokens = (
        genres   * 3 +
        [director] * 3 +
        cast     * 1 +
        keywords * 1
    )

    return " ".join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def fit_vectorizer(all_soups: list) -> tuple:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_soups)
    return vectorizer, matrix

def build_taste_vector(liked_vectors, star_ratings: list) -> np.ndarray:
    dense = liked_vectors.toarray()
    weights = np.array(star_ratings)
    taste = np.average(dense, axis=0, weights=weights)
    return taste.reshape(1, -1)

from src.tmdb import tmdb_get
from src.cache import cached_features

def get_candidate_pool(liked_features: list, limit: int = 300) -> list:
    candidates = {}  


    all_genres = []
    for f in liked_features:
        for g in f["genres"]:
            if g not in all_genres:
                all_genres.append(g)

    genre_data = tmdb_get("/genre/movie/list")
    genre_map  = {g["name"]: g["id"] for g in genre_data.get("genres", [])}
    genre_ids  = [str(genre_map[g]) for g in all_genres if g in genre_map]

    for page in range(1, 6): 
        data = tmdb_get("/discover/movie", params={
            "with_genres":    ",".join(genre_ids),
            "sort_by":        "popularity.desc",
            "vote_count.gte": 100,
            "page":           page
        })
        for movie in data.get("results", []):
            mid = movie["id"]
            if mid not in candidates:
                candidates[mid] = cached_features(mid)

    top_liked = liked_features[:3]
    for f in top_liked:
        data = tmdb_get(f"/movie/{f['id']}/recommendations")
        for movie in data.get("results", []):
            mid = movie["id"]
            if mid not in candidates:
                candidates[mid] = cached_features(mid)

    return list(candidates.values())[:limit]

from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(taste_vector, candidate_vectors, candidates: list) -> list:
    sims = cosine_similarity(taste_vector, candidate_vectors)[0]

    for i, candidate in enumerate(candidates):
        candidate["score"] = float(sims[i])

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return ranked

def recommend_from_features(liked_features: list, star_ratings: list, candidate_pool: list) -> list:
    liked_soups     = [build_soup(f) for f in liked_features]
    candidate_soups = [build_soup(f) for f in candidate_pool]
    all_soups       = liked_soups + candidate_soups

    vectorizer, all_matrix = fit_vectorizer(all_soups)

    liked_matrix     = all_matrix[:len(liked_soups)]
    candidate_matrix = all_matrix[len(liked_soups):]

    taste_vector = build_taste_vector(liked_matrix, star_ratings)

    return rank_candidates(taste_vector, candidate_matrix, candidate_pool)

def recommend_for_user(liked_features: list, star_ratings: list, candidate_pool: list) -> list:
    ranked = recommend_from_features(liked_features, star_ratings, candidate_pool)

    watched = set()
    for f in liked_features:
        title = f["title"].lower().strip()
        watched.add(title)

    unseen = []
    for film in ranked:
        title = film["title"].lower().strip()
        if title not in watched:
            unseen.append(film)

    return unseen


from src.letterboxd import get_taste_profile
from src.tmdb import search_movie

def recommend_for_letterboxd_user(username: str, top_n: int = 10) -> list:
    profile = get_taste_profile(username)
    if not profile:
        return []

    all_titles = []
    seen_titles = set()
    for title in profile["recent_films"] + profile["highly_rated"]:
        if title not in seen_titles:
            all_titles.append(title)
            seen_titles.add(title)

    liked_features = []
    star_ratings   = []
    for title in all_titles[:20]:  
        result = search_movie(clean_title(title))
        if result:
            features = cached_features(result["id"])
            liked_features.append(features)
            star_ratings.append(4.0)  
    if not liked_features:
        return []

    pool = get_candidate_pool(liked_features)
    return recommend_for_user(liked_features, star_ratings, pool)[:top_n]