"""Build the corpus, fit the model, and tune its weights.

    python -m src.train --movies 10000       # fetch corpus, fit, save
    python -m src.train --fit-only           # refit from whatever is cached
    python -m src.train --tune dave          # grid-search the soup weights

"Training" here is fitting TF-IDF over a fixed corpus and choosing the
soup weights that score best on leave-one-out. There are no learned
parameters beyond the vocabulary and IDF statistics - which is exactly
why the weights are worth tuning against a real metric rather than
guessed at.
"""
import argparse
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import cache, config, model as model_mod
from src.taste import build_soup, fit_vectorizer, soup_tokens
from src.tmdb import tmdb_get

log = logging.getLogger(__name__)


# TMDB genre ids. Sweeping per-genre is what stops the corpus being 90%
# blockbusters: Documentary, Western and History barely appear in a plain
# popularity sort, so nothing in those genres was ever recommendable.
GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Thriller": 53, "War": 10752, "Western": 37,
}


def _discover_pages(params: dict, pages: int, label: str, workers: int = 12) -> list[int]:
    """Pull up to `pages` pages of /discover, fetched concurrently.

    Page 1 goes first and alone, both to seed the id list and to read
    TMDB's own total_pages - a thin sweep (Western, 1920s) might have
    three pages of qualifying films, and there is no point spinning up a
    pool of workers to discover that. Whatever is left is fanned out.
    """
    def _page(n: int) -> list[int]:
        try:
            data = tmdb_get("/discover/movie",
                            params={**params, "include_adult": "false", "page": n})
        except Exception as exc:
            log.warning("discover [%s] page %d failed: %s", label, n, exc)
            return [], 1
        return [m["id"] for m in data.get("results", [])], data.get("total_pages", 1)

    ids, total_pages = _page(1)
    last_page = min(pages, total_pages or 1)
    if last_page <= 1 or not ids:
        return ids

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_page, n) for n in range(2, last_page + 1)]
        for future in as_completed(futures):
            page_ids, _ = future.result()
            ids.extend(page_ids)
    return ids



# Non-English markets big enough on TMDB to be worth a dedicated sweep.
# The other sweeps sort globally by popularity/vote count, which surfaces
# only the handful of foreign films that crossed over (Parasite, Your
# Name) - everything one tier below that was invisible.
LANGUAGE_SWEEPS = ["ja", "ko", "fr", "es", "hi", "de", "it", "zh",
                   "ru", "pt", "sv", "da"]


def discover_movie_ids(target: int, min_votes: int = 200) -> list[int]:
    """Ids from several complementary sweeps, not just "popular right now".

    A single popularity sort gave a corpus that was recent, English and
    mainstream: Moon, Sunshine and Ad Astra were all missing while three
    Twilight films were present. Recommending "something like Interstellar"
    from that corpus cannot work however good the ranking is, so the
    corpus is built from several sweeps instead - each raised well past
    where it stops finding anything, since a sweep that runs dry just
    returns fewer ids rather than wasting the extra page budget.
    """
    sweeps = []

    # 1. What people are actually watching.
    sweeps.append(("popular", {"sort_by": "popularity.desc",
                               "vote_count.gte": min_votes}, 80))

    # 2. What is actually good, regardless of current buzz.
    sweeps.append(("acclaimed", {"sort_by": "vote_average.desc",
                                 "vote_count.gte": 500}, 80))

    # 3. Decade by decade, so the corpus is not all post-2010. Pushed back
    # to the 1920s for the silent/early-sound era, with a lower vote-count
    # bar early on since fewer people rate a 1928 film than a 2019 one.
    for start in range(1920, 2030, 10):
        sweeps.append((f"{start}s", {
            "sort_by": "vote_count.desc",
            "vote_count.gte": 30 if start < 1960 else 100,
            "primary_release_date.gte": f"{start}-01-01",
            "primary_release_date.lte": f"{start + 9}-12-31",
        }, 25))

    # 4. Genre by genre, so thin genres are represented at all.
    for name, genre_id in GENRE_IDS.items():
        sweeps.append((f"genre:{name}", {
            "sort_by": "vote_count.desc",
            "vote_count.gte": 50,
            "with_genres": str(genre_id),
        }, 35))

    # 5. Language by language, so the corpus isn't Hollywood-only.
    for lang in LANGUAGE_SWEEPS:
        sweeps.append((f"lang:{lang}", {
            "sort_by": "vote_count.desc",
            "vote_count.gte": 80,
            "with_original_language": lang,
        }, 25))

    ids = []
    for label, params, pages in sweeps:
        found = _discover_pages(params, pages, label)
        log.info("  sweep %-18s -> %4d ids", label, len(found))
        ids.extend(found)
        if len(dict.fromkeys(ids)) >= target:
            break

    return list(dict.fromkeys(ids))[:target]


def build_corpus(target: int, workers: int = 8, refresh_stale: bool = True) -> int:
    """Fetch and cache features for the top `target` movies.

    Rows written by an older feature schema are refetched too - a thin row
    sitting next to rich ones is worse than no row, because its short
    vector wins cosine comparisons it should lose.
    """
    ids = discover_movie_ids(target)
    todo = [i for i in ids if not cache.is_current(cache.get_cached(i))]

    if refresh_stale:
        known = set(ids)
        todo += [i for i in cache.stale_ids() if i not in known]

    todo = list(dict.fromkeys(todo))
    log.info("corpus target %d movies - %d to fetch or refresh",
             len(ids), len(todo))

    if not todo:
        return 0

    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cache.cached_features, i): i for i in todo}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                log.warning("movie %s failed: %s", futures[future], exc)
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - started)
                log.info("  fetched %d/%d (%.1f/s)", done, len(todo), rate)

    return done


def fit(weights: dict = None, metrics: dict = None) -> model_mod.TasteModel:
    weights = weights or config.SOUP_WEIGHTS
    movies = [m for m in cache.all_movies() if m.get("genres")]
    if not movies:
        raise RuntimeError("Cache is empty - run: python -m src.train")

    # A row with almost no tokens puts its entire L2 norm on a handful of
    # terms, so it wins cosine comparisons against far better matches.
    # That is precisely how 23 sparse documentaries came to outrank the
    # whole corpus. Drop them rather than let them distort the ranking.
    keep = [m for m in movies if soup_tokens(m, weights) >= config.MIN_SOUP_TOKENS]
    dropped = len(movies) - len(keep)
    if dropped:
        log.info("dropped %d movies with fewer than %d tokens",
                 dropped, config.MIN_SOUP_TOKENS)
    movies = keep

    soups = [build_soup(m, weights) for m in movies]
    vectorizer, matrix = fit_vectorizer(soups)

    log.info("fitted TF-IDF on %d movies, %d terms", len(movies), len(vectorizer.vocabulary_))
    return model_mod.TasteModel(
        vectorizer=vectorizer, matrix=matrix, movies=movies,
        weights=dict(weights), metrics=metrics or {},
    )


def tune(usernames: list[str], max_films: int = 20) -> tuple[dict, dict]:
    """Grid-search soup weights against mean leave-one-out MRR.

    Tune against profiles with focused taste, and ideally more than one.
    A profile with no coherent thread through it (someone who watches a bit
    of everything) scores near random whatever the weights are, so tuning
    on it just fits noise.
    """
    from src.evaluate import leave_one_out
    from src.letterboxd import get_taste_profile

    profiles = []
    for username in usernames:
        profile = get_taste_profile(username)
        if not profile:
            log.warning("could not fetch Letterboxd profile for %r - skipping", username)
            continue
        profiles.append(profile)

    if not profiles:
        raise RuntimeError("No usable Letterboxd profiles to tune against")

    # Only the four load-bearing fields are searched; the rest keep their
    # configured defaults. Merging with SOUP_WEIGHTS matters - a grid dict
    # that omitted the newer fields would quietly refit the model without
    # overview terms, which is the very sparsity problem they exist to fix.
    grid = [
        {**config.SOUP_WEIGHTS,
         "genres": g, "director": d, "cast": c, "keywords": k}
        for g, d, c, k in itertools.product((1, 3), (2, 3, 5), (1, 2), (1, 2, 3))
    ]

    best, best_metrics = None, None
    for i, weights in enumerate(grid, start=1):
        candidate = fit(weights)

        folds = [leave_one_out(profile=p, taste_model=candidate,
                               weights=weights, max_films=max_films)
                 for p in profiles]
        folds = [f for f in folds if f]
        if not folds:
            continue

        metrics = {
            "mrr":         round(sum(f["mrr"] for f in folds) / len(folds), 4),
            "hit_at_10":   round(sum(f["hit_at_10"] for f in folds) / len(folds), 3),
            "median_rank": round(sum(f["median_rank"] for f in folds) / len(folds), 1),
            "profiles":    len(folds),
        }
        log.info("[%2d/%d] %s -> MRR %.4f  hit@10 %.2f  median %.0f",
                 i, len(grid), weights, metrics["mrr"],
                 metrics["hit_at_10"], metrics["median_rank"])

        if best_metrics is None or metrics["mrr"] > best_metrics["mrr"]:
            best, best_metrics = weights, metrics

    return best, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the movie-bot model")
    parser.add_argument("--movies", type=int, default=10000,
                        help="how many movies to have in the corpus")
    parser.add_argument("--fit-only", action="store_true",
                        help="skip fetching; refit from the existing cache")
    parser.add_argument("--tune", metavar="USERNAME", nargs="+",
                        help="grid-search soup weights against these Letterboxd users")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.require("TMDB_TOKEN")

    if not args.fit_only:
        build_corpus(args.movies, workers=args.workers)
    log.info("cache now holds %d movies (%d on the current schema)",
             cache.count(), cache.count(current_only=True))

    weights, metrics = config.SOUP_WEIGHTS, {}
    if args.tune:
        weights, metrics = tune(args.tune)
        log.info("best weights: %s -> %s", weights, metrics)

    model_mod.save(fit(weights, metrics))
    print(f"\nModel saved to {config.MODEL_FILE}")
    print(f"  corpus:  {cache.count()} movies")
    print(f"  weights: {weights}")
    if metrics:
        print(f"  metrics: {metrics}")


if __name__ == "__main__":
    main()
