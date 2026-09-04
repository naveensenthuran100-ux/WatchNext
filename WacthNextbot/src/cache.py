import json
import os
from src.tmdb import get_movie_features

CACHE_FILE = "data/features_cache.json"
#_load_cache the undersdcore is just to show that this is private, just convention

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

cache = _load_cache()

def cached_features(movie_id: int) -> dict:
    key = str(movie_id)

    if key in cache:
        return cache[key]

    print(f"[cache] MISS — fetching {movie_id} from TMDB")
    features = get_movie_features(movie_id)

    cache[key] = features
    _save_cache(cache)

    return features