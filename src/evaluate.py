"""Leave-one-out evaluation.

For each film a user liked: hide it, build a taste vector from the rest,
rank the whole corpus, and see where the hidden film lands. If the model
captures taste at all, hidden favourites should rank near the top.

This is the objective `python -m src.train --tune` optimises against.
"""
import argparse
import logging
import statistics

import numpy as np

from scipy import sparse

from src import config, model as model_mod
from src.letterboxd import get_taste_profile
from src.taste import build_soup, build_taste_vector, rank_candidates, resolve_films

log = logging.getLogger(__name__)


def _build_eval_corpus(taste_model, user_features, weights):
    """Corpus matrix + movie list, guaranteed to contain the user's films."""
    movies = list(taste_model.movies)
    matrix = taste_model.matrix
    known  = {m["id"] for m in movies}

    missing = [f for f in user_features if f["id"] not in known]
    if missing:
        extra = taste_model.vectorizer.transform([build_soup(f, weights) for f in missing])
        matrix = sparse.vstack([matrix, extra]).tocsr()
        movies = movies + missing

    return movies, matrix


def leave_one_out(username: str = None, profile: dict = None, taste_model=None,
                  weights: dict = None, max_films: int = 25) -> dict:
    taste_model = taste_model or model_mod.load()
    if taste_model is None:
        raise RuntimeError("No trained model. Run: python -m src.train")

    weights = weights or taste_model.weights or config.SOUP_WEIGHTS

    if profile is None:
        profile = get_taste_profile(username)
    if not profile:
        return {}

    features, ratings = resolve_films(profile["liked"][:max_films])
    if len(features) < 3:
        log.warning("only %d films resolved - not enough to evaluate", len(features))
        return {}

    movies, matrix = _build_eval_corpus(taste_model, features, weights)
    index_by_id = {m["id"]: i for i, m in enumerate(movies)}

    user_matrix = taste_model.vectorizer.transform([build_soup(f, weights) for f in features])

    ranks = []
    for i, hidden in enumerate(features):
        keep_rows = [r for r in range(user_matrix.shape[0]) if r != i]
        rest_ratings = [ratings[r] for r in keep_rows]
        taste_vector = build_taste_vector(user_matrix[keep_rows], rest_ratings)

        # Rank everything except the other films we already know they've seen.
        exclude = {features[r]["id"] for r in keep_rows}
        rows    = [j for j, m in enumerate(movies) if m["id"] not in exclude]

        ranked = rank_candidates(taste_vector, matrix[rows], [movies[j] for j in rows])

        for position, film in enumerate(ranked, start=1):
            if film["id"] == hidden["id"]:
                ranks.append(position)
                break

    if not ranks:
        return {}

    n = len(ranks)
    return {
        "n":            n,
        "corpus":       len(movies),
        "hit_at_10":    round(sum(r <= 10 for r in ranks) / n, 3),
        "hit_at_50":    round(sum(r <= 50 for r in ranks) / n, 3),
        "median_rank":  round(statistics.median(ranks), 1),
        "mrr":          round(sum(1.0 / r for r in ranks) / n, 4),
    }


def sanity_check(taste_model=None) -> list[dict]:
    """Offline check that the ranking machinery works at all.

    Builds synthetic users with genuinely coherent taste (every film by one
    director, or one genre pair), hides one film, and ranks the corpus. A
    working model puts the hidden film near rank 1; anything close to
    corpus/2 means it is guessing.

    This exists because a poor score on a real profile is ambiguous - it
    can mean the model is broken, or just that the person's taste is not
    predictable from content. This separates the two.
    """
    taste_model = taste_model or model_mod.load()
    if taste_model is None:
        raise RuntimeError("No trained model. Run: python -m src.train")

    movies = taste_model.movies
    matrix = taste_model.matrix
    ids    = np.array([m["id"] for m in movies])

    cohorts = [
        ("Nolan films",      lambda m: m.get("director") == "Christopher Nolan"),
        ("Miyazaki films",   lambda m: m.get("director") == "Hayao Miyazaki"),
        ("Horror+Thriller",  lambda m: {"Horror", "Thriller"} <= set(m["genres"])),
        ("Animation+Family", lambda m: {"Animation", "Family"} <= set(m["genres"])),
        ("Documentary",      lambda m: "Documentary" in m["genres"]),
    ]

    results = []
    for label, predicate in cohorts:
        rows = [i for i, m in enumerate(movies) if predicate(m)]
        if len(rows) < 5:
            continue

        cohort_matrix = matrix[rows]
        ranks = []
        for i in range(len(rows)):
            keep    = [r for r in range(len(rows)) if r != i]
            exclude = {movies[rows[r]]["id"] for r in keep}
            cand    = [j for j, m in enumerate(movies) if m["id"] not in exclude]

            taste_vector = build_taste_vector(cohort_matrix[keep], [4.0] * len(keep))
            ranked = rank_candidates(taste_vector, matrix[cand], [movies[j] for j in cand])

            target = movies[rows[i]]["id"]
            for position, film in enumerate(ranked, start=1):
                if film["id"] == target:
                    ranks.append(position)
                    break

        results.append({
            "cohort":      label,
            "n":           len(ranks),
            "median_rank": round(statistics.median(ranks), 1),
            "random_rank": len(movies) // 2,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation")
    parser.add_argument("username", nargs="?", help="Letterboxd username")
    parser.add_argument("--sanity", action="store_true",
                        help="offline check that the model ranks coherent taste correctly")
    parser.add_argument("--max-films", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.sanity:
        print("\n--- Sanity check (synthetic coherent tastes) ---")
        for row in sanity_check():
            verdict = "OK" if row["median_rank"] <= row["random_rank"] / 10 else "WEAK"
            print(f"  {row['cohort']:18} n={row['n']:3}  median rank "
                  f"{row['median_rank']:6}  (random ~{row['random_rank']})  {verdict}")
        return

    if not args.username:
        parser.error("give a username, or use --sanity")

    results = leave_one_out(args.username, max_films=args.max_films)
    if not results:
        print("Could not evaluate - check the username, or that the profile has enough rated films.")
        return

    print("\n--- Results ---")
    print(f"films tested: {results['n']}  (corpus of {results['corpus']})")
    print(f"hit@10:       {results['hit_at_10']}  ({results['hit_at_10']*100:.0f}% of hidden films in the top 10)")
    print(f"hit@50:       {results['hit_at_50']}")
    print(f"median rank:  {results['median_rank']}")
    print(f"MRR:          {results['mrr']}")


if __name__ == "__main__":
    main()
