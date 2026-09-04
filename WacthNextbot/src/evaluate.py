# src/evaluate.py

import re
from src.tmdb import search_movie
from src.cache import cached_features
from src.taste import (
    build_soup, fit_vectorizer, build_taste_vector,
    get_candidate_pool, rank_candidates
)
from src.letterboxd import get_taste_profile

def clean_title(title: str) -> str:
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

def leave_one_out(username: str) -> dict:
    profile = get_taste_profile(username)
    if not profile:
        return {}

    all_titles = []
    seen = set()
    for title in profile["recent_films"] + profile["highly_rated"]:
        if title not in seen:
            all_titles.append(title)
            seen.add(title)

    all_features = []
    for title in all_titles[:20]:
        result = search_movie(clean_title(title))
        if result:
            features = cached_features(result["id"])
            all_features.append(features)

    if len(all_features) < 3:
        print("Not enough films to evaluate")
        return {}

    print(f"Evaluating {len(all_features)} films for {username}...")

    hits     = 0   
    ranks    = []  

    for i in range(len(all_features)):
        hidden   = all_features[i]
        the_rest = all_features[:i] + all_features[i+1:]
        ratings  = [4.0] * len(the_rest)

        pool = get_candidate_pool(the_rest, limit=100)

        if not any(c["id"] == hidden["id"] for c in pool):
            pool.append(hidden)

        rest_soups      = [build_soup(f) for f in the_rest]
        candidate_soups = [build_soup(f) for f in pool]
        all_soups       = rest_soups + candidate_soups

        vectorizer, all_matrix = fit_vectorizer(all_soups)

        rest_matrix      = all_matrix[:len(rest_soups)]
        candidate_matrix = all_matrix[len(rest_soups):]

        taste_vector = build_taste_vector(rest_matrix, ratings)
        ranked       = rank_candidates(taste_vector, candidate_matrix, pool)

        rank = None
        for position, film in enumerate(ranked):
            if film["id"] == hidden["id"]:
                rank = position + 1 
                break

        if rank is not None:
            ranks.append(rank)
            if rank <= 10:
                hits += 1
            print(f"  [{i+1}/{len(all_features)}] '{hidden['title']}' ranked #{rank}")

    n        = len(ranks)
    hit_at_10 = hits / n if n > 0 else 0
    median_rank = sorted(ranks)[n // 2] if ranks else None

    return {
        "hit_at_10":   round(hit_at_10, 3),
        "median_rank": median_rank,
        "n":           n
    }

if __name__ == "__main__":
    results = leave_one_out("dave")
    print("\n--- Results ---")
    print(f"hit@10:      {results['hit_at_10']} ({results['hit_at_10']*100:.1f}% of films landed in top 10)")
    print(f"median rank: {results['median_rank']}")
    print(f"films tested:{results['n']}")