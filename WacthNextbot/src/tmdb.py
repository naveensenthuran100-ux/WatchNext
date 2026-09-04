import requests
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_HEADERS = {"Authorization": f"Bearer {os.getenv('TMDB_TOKEN')}"}

def tmdb_get(path: str, params: dict = None) -> dict:
    url = f"{TMDB_BASE}{path}"
    response = requests.get(url, headers=TMDB_HEADERS, params=params)
    return response.json()

def search_movie(title: str) -> dict | None:
    data = tmdb_get("/search/movie", params={"query": title})
    results = data.get("results", [])
    return results[0] if results else None

def get_movie_features(movie_id: int) -> dict:
    details  = tmdb_get(f"/movie/{movie_id}")
    keywords = tmdb_get(f"/movie/{movie_id}/keywords")
    credits  = tmdb_get(f"/movie/{movie_id}/credits")

    genres = []
    for g in details.get("genres", []):
        genres.append(g["name"])

    kw_list = [k["name"] for k in keywords.get("keywords", [])]

    cast = [c["name"] for c in credits.get("cast", [])[:3]]

    director = ""
    for person in credits.get("crew", []):
        if person["job"] == "Director":
            director = person["name"]
            break

    return {
        "id":           movie_id,
        "title":        details.get("title", ""),
        "year":         details.get("release_date", "")[:4],
        "genres":       genres,
        "keywords":     kw_list[:10], 
        "cast":         cast,
        "director":     director,
        "overview":     details.get("overview", ""),
        "vote_average": details.get("vote_average", 0.0),
    }


