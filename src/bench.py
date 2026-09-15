"""A/B benchmark for scoring signals.

Three metrics, because the bot does three different jobs and a change can
easily help one while wrecking another:

  reference  - "something like X". Seed a known film, see where films a
               human would obviously accept actually rank. This is the
               headline use case and the one that was broken.
  cohort     - synthetic coherent tastes (all Nolan, all Miyazaki). A
               sanity check on the machinery: if these do not rank near
               the top, the ranking is simply wrong.
  profile    - leave-one-out on a real Letterboxd account. The honest,
               noisy, hardest metric.

Run:  python -m src.bench                  # every signal, plus the blend
      python -m src.bench --profile ishita # include the profile metric
"""
import argparse
import logging
import statistics

import numpy as np

from src import config, corpus

log = logging.getLogger(__name__)


# Seed film -> films most people would accept as "like it". Deliberately
# obvious: the point is to catch a ranking that is wrong, not to referee
# arguable taste calls.
REFERENCE_CASES = {
    "Interstellar": ["Arrival", "Contact", "The Martian", "Gravity",
                     "2001: A Space Odyssey", "Ad Astra", "Moon", "Sunshine",
                     "Apollo 13", "First Man"],
    "Heat":         ["Collateral", "The Town", "Sicario", "Ronin",
                     "Point Break", "The Score", "Public Enemies"],
    "The Shining":  ["The Exorcist", "Rosemary's Baby", "Hereditary",
                     "The Witch", "Doctor Sleep", "Don't Look Now"],
    "Toy Story":    ["Toy Story 2", "Toy Story 3", "Monsters, Inc.",
                     "Finding Nemo", "A Bug's Life", "WALL·E"],
    "Alien":        ["Aliens", "Event Horizon", "The Thing", "Prometheus",
                     "Blade Runner", "Alien³"],
    "Die Hard":     ["Lethal Weapon", "Speed", "Con Air", "The Rock",
                     "Under Siege", "Predator"],
    "Amélie":       ["The Grand Budapest Hotel", "Delicatessen",
                     "Chocolat", "Midnight in Paris"],
    "Get Out":      ["Us", "Nope", "The Invitation", "Hereditary",
                     "Midsommar", "Don't Breathe"],
    "La La Land":   ["Whiplash", "Singin' in the Rain", "Moulin Rouge!",
                     "The Artist", "Babylon"],
    "Mad Max: Fury Road": ["The Road Warrior", "Dredd", "Edge of Tomorrow",
                           "Snowpiercer", "Death Race"],
}


# Free-text requests, of the kind people actually send the bot. Each case
# carries the keywords a good LLM extraction would produce, so the
# metadata signal is measured at its best rather than handed raw prose it
# was never designed to read. Semantic gets the text; metadata gets the
# keywords; both are then scored identically.
QUERY_CASES = [
    {"text": "a stranded astronaut struggling to survive alone in space",
     "keywords": ["space", "astronaut", "survival", "isolation"],
     "expect": ["The Martian", "Gravity", "Stowaway", "Moon", "Ad Astra",
                "Apollo 13"]},
    {"text": "a slow burn psychological horror about isolation and madness",
     "keywords": ["psychological horror", "isolation", "madness"],
     "expect": ["The Shining", "Hereditary", "The Lighthouse", "Midsommar",
                "The Witch"]},
    {"text": "a heist that goes wrong and the crew turns on each other",
     "keywords": ["heist", "robbery", "betrayal", "crime"],
     "expect": ["Reservoir Dogs", "Heat", "Dog Day Afternoon", "The Killing",
                "Inside Man"]},
    {"text": "a tender coming of age story about first love",
     "keywords": ["coming of age", "first love", "adolescence"],
     "expect": ["Call Me by Your Name", "Lady Bird", "Boyhood", "Stand by Me"]},
    {"text": "a mind bending puzzle about memory and identity",
     "keywords": ["memory", "identity", "nonlinear", "mind bending"],
     "expect": ["Memento", "Inception", "Shutter Island", "Mulholland Drive",
                "Primer"]},
    {"text": "a courtroom drama about justice and a wrongful conviction",
     "keywords": ["courtroom", "trial", "justice", "wrongful conviction"],
     "expect": ["12 Angry Men", "A Few Good Men", "Just Mercy", "Philadelphia"]},
    {"text": "a colourful animated adventure with talking animals for kids",
     "keywords": ["animation", "talking animals", "family", "adventure"],
     "expect": ["Zootopia", "The Lion King", "Madagascar", "Finding Nemo"]},
    {"text": "a bleak war film about the futility of trench warfare",
     "keywords": ["world war i", "trench warfare", "war", "soldiers"],
     "expect": ["1917", "All Quiet on the Western Front", "Paths of Glory",
                "Dunkirk"]},
    {"text": "a bittersweet romance about ambition pulling two people apart",
     "keywords": ["romance", "ambition", "musician", "bittersweet"],
     "expect": ["La La Land", "Before Sunrise", "Whiplash", "Amadeus"]},
]


def query_quality(signal) -> dict:
    """Where do the right films rank for a request typed in plain English?

    This is the metric the metadata signal structurally cannot win. Its
    vocabulary has no entry for "mind bending", so a free-text request
    has to be translated into keyword tokens that then have to literally
    appear on the right films. Embedding the sentence skips both steps.
    """
    titles = corpus.title_index()
    movies = corpus.movies()

    ranks, found, expected_total = [], 0, 0
    for case in QUERY_CASES:
        query = signal.encode_query(case["text"], case["keywords"])
        sims = signal.similarity(query)
        if sims is None:
            continue

        order = np.argsort(sims)[::-1]
        position = {movies[j]["id"]: r for r, j in enumerate(order, start=1)}

        for title in case["expect"]:
            expected_total += 1
            j = titles.get(title.lower())
            if j is None or not signal.covered[j]:
                continue
            found += 1
            ranks.append(position[movies[j]["id"]])

    if not ranks:
        return {}
    n = len(ranks)
    return {
        "n":           n,
        "in_corpus":   f"{found}/{expected_total}",
        "median_rank": round(statistics.median(ranks), 1),
        "hit_at_10":   round(sum(r <= 10 for r in ranks) / n, 3),
        "hit_at_50":   round(sum(r <= 50 for r in ranks) / n, 3),
        "mrr":         round(sum(1.0 / r for r in ranks) / n, 4),
    }


def reference_quality(signal) -> dict:
    """Where do obviously-similar films rank when seeded with one film?"""
    titles = corpus.title_index()
    movies = corpus.movies()

    ranks, found, expected_total = [], 0, 0
    for seed, expected in REFERENCE_CASES.items():
        i = titles.get(seed.lower())
        if i is None or not signal.covered[i]:
            continue

        sims = signal.similarity_to_row(i)
        if sims is None:
            continue
        sims[i] = -np.inf                      # never rank the seed itself

        order = np.argsort(sims)[::-1]
        position = {movies[j]["id"]: r for r, j in enumerate(order, start=1)}

        for title in expected:
            expected_total += 1
            j = titles.get(title.lower())
            if j is None or not signal.covered[j]:
                continue                       # not in the corpus: not a miss
            found += 1
            ranks.append(position[movies[j]["id"]])

    if not ranks:
        return {}
    n = len(ranks)
    return {
        "n":           n,
        "in_corpus":   f"{found}/{expected_total}",
        "median_rank": round(statistics.median(ranks), 1),
        "hit_at_10":   round(sum(r <= 10 for r in ranks) / n, 3),
        "hit_at_50":   round(sum(r <= 50 for r in ranks) / n, 3),
        "mrr":         round(sum(1.0 / r for r in ranks) / n, 4),
    }


COHORTS = [
    ("Nolan",       lambda m: m.get("director") == "Christopher Nolan"),
    ("Miyazaki",    lambda m: m.get("director") == "Hayao Miyazaki"),
    ("Horror+Thrl", lambda m: {"Horror", "Thriller"} <= set(m.get("genres", []))),
    ("Anim+Family", lambda m: {"Animation", "Family"} <= set(m.get("genres", []))),
]


def cohort_quality(signal) -> dict:
    """Hide one film from a coherent set; can the rest find it?"""
    movies = corpus.movies()
    medians = []
    for _, predicate in COHORTS:
        rows = [i for i, m in enumerate(movies)
                if predicate(m) and signal.covered[i]]
        if len(rows) < 5:
            continue

        ranks = []
        for i in rows:
            others = [r for r in rows if r != i]
            scores = signal.profile_similarity_rows(others)
            if scores is None:
                continue
            scores[others] = -np.inf           # exclude the films we showed it
            ranks.append(int((scores > scores[i]).sum()) + 1)
        if ranks:
            medians.append(statistics.median(ranks))

    if not medians:
        return {}
    return {"median_rank": round(statistics.mean(medians), 1),
            "cohorts": len(medians)}


def _usable(vector) -> bool:
    """Is this query vector worth anything?

    A blend returns a dict of per-signal vectors, a plain signal returns
    an array. Coercing the dict with np.asarray would wrap it in an
    object array and quietly defeat every isinstance check downstream,
    so the two shapes are handled explicitly here instead.
    """
    if vector is None:
        return False
    if isinstance(vector, dict):
        return any(_usable(v) for v in vector.values())
    return bool(np.asarray(vector).any())


# Pairs of genuinely disjoint tastes. A single centroid over both lands
# in the empty space between them, so this is the metric that separates a
# user model that handles real, mixed taste from one that only works when
# somebody watches exactly one kind of film.
MIXED_PAIRS = [
    (lambda m: {"Animation", "Family"} <= set(m.get("genres", [])),
     lambda m: {"Horror", "Thriller"} <= set(m.get("genres", []))),
    (lambda m: "Documentary" in m.get("genres", []),
     lambda m: {"Action", "Adventure"} <= set(m.get("genres", []))),
    (lambda m: {"Romance", "Comedy"} <= set(m.get("genres", [])),
     lambda m: "War" in m.get("genres", [])),
]


def mixed_cohort_quality(signal, per_side: int = 8) -> dict:
    """Can it serve a user who likes two unrelated kinds of film?"""
    movies = corpus.movies()

    def rows_where(predicate):
        return [i for i, m in enumerate(movies)
                if predicate(m) and signal.covered[i]][:25]

    ranks = []
    for first, second in MIXED_PAIRS:
        a, b = rows_where(first), rows_where(second)
        if len(a) < 6 or len(b) < 6:
            continue
        for target_pool, other in ((a, b), (b, a)):
            for i in target_pool[:per_side]:
                liked = [r for r in target_pool if r != i] + other
                scores = signal.profile_similarity_rows(liked)
                if scores is None:
                    continue
                scores[liked] = -np.inf
                ranks.append(int((scores > scores[i]).sum()) + 1)

    if not ranks:
        return {}
    n = len(ranks)
    return {"n": n,
            "median_rank": round(statistics.median(ranks), 1),
            "hit_at_10": round(sum(r <= 10 for r in ranks) / n, 3)}


def profile_quality(signal, username: str, max_films: int = 25) -> dict:
    """Leave-one-out against a real Letterboxd account."""
    from src.recommender import load_profile
    from src.taste import resolve_films

    profile = load_profile(username)
    if not profile:
        return {}

    features, ratings = resolve_films(profile["liked"][:max_films])
    vectors, kept_ratings, targets = [], [], []
    for f, r in zip(features, ratings):
        row = corpus.row_of(f["id"])
        vector = signal.encode_movie(f)
        if not _usable(vector):
            continue
        vectors.append(vector)
        kept_ratings.append(r)
        targets.append(row)

    if len(vectors) < 3:
        return {}

    ranks = []
    for i, row in enumerate(targets):
        if row is None:
            continue
        others = [v for j, v in enumerate(vectors) if j != i]
        other_ratings = [r for j, r in enumerate(kept_ratings) if j != i]
        scores = signal.profile_similarity(others, other_ratings)
        if scores is None:
            continue
        seen = [t for j, t in enumerate(targets) if j != i and t is not None]
        scores[seen] = -np.inf
        ranks.append(int((scores > scores[row]).sum()) + 1)

    if not ranks:
        return {}
    n = len(ranks)
    return {
        "n":           n,
        "median_rank": round(statistics.median(ranks), 1),
        "hit_at_10":   round(sum(r <= 10 for r in ranks) / n, 3),
        "hit_at_50":   round(sum(r <= 50 for r in ranks) / n, 3),
        "mrr":         round(sum(1.0 / r for r in ranks) / n, 4),
    }


def evaluate(signal, username: str = None) -> dict:
    result = {"name": signal.name, "coverage": round(signal.coverage, 3),
              "reference": reference_quality(signal),
              "query": query_quality(signal),
              "cohort": cohort_quality(signal),
              "mixed": mixed_cohort_quality(signal)}
    if username:
        result["profile"] = profile_quality(signal, username)
    return result


def print_table(results: list) -> None:
    header = (f"{'signal':<12} {'cov':>5} | {'ref med':>8} {'ref@10':>7} "
              f"{'ref MRR':>8} | {'qry med':>8} {'qry@10':>7} {'qry MRR':>8} | "
              f"{'cohort':>7} {'mixed':>7} | {'prof med':>9} {'prof MRR':>9}")
    print(header)
    print("-" * len(header))
    nan = float("nan")
    for r in results:
        ref = r.get("reference", {})
        qry = r.get("query", {})
        coh = r.get("cohort", {})
        mix = r.get("mixed", {})
        prof = r.get("profile", {})
        print(f"{r['name']:<12} {r['coverage']:>5.2f} | "
              f"{ref.get('median_rank', nan):>8.1f} "
              f"{ref.get('hit_at_10', nan):>7.3f} "
              f"{ref.get('mrr', nan):>8.4f} | "
              f"{qry.get('median_rank', nan):>8.1f} "
              f"{qry.get('hit_at_10', nan):>7.3f} "
              f"{qry.get('mrr', nan):>8.4f} | "
              f"{coh.get('median_rank', nan):>7.1f} "
              f"{mix.get('median_rank', nan):>7.1f} | "
              f"{prof.get('median_rank', nan):>9.1f} "
              f"{prof.get('mrr', nan):>9.4f}")


def build_all(include=None):
    """Every signal that is currently available, plus the blend."""
    from src import model as model_mod, signals as sig

    built = []
    include = include or {"metadata", "semantic", "graph", "collab", "blend"}

    taste_model = model_mod.load()
    if taste_model and "metadata" in include:
        built.append(sig.MetadataSignal(taste_model))

    if "semantic" in include:
        semantic = sig.load_semantic()
        if semantic is None:
            log.warning("no embeddings yet - run: python -m src.embed")
        else:
            built.append(semantic)

    if "graph" in include:
        try:
            from src.graph import load_graph
            graph = load_graph()
            if graph is None:
                log.warning("no graph signal yet - run: python -m src.graph")
            else:
                built.append(graph)
        except ImportError:
            pass

    if "collab" in include:
        try:
            from src.collab import load_collab
            collab = load_collab()
            if collab is None:
                log.warning("no collaborative model yet - run: python -m src.collab")
            else:
                built.append(collab)
        except ImportError:
            pass

    return built


def main():
    parser = argparse.ArgumentParser(description="Benchmark scoring signals")
    parser.add_argument("--profile", help="Letterboxd username for the LOO metric")
    parser.add_argument("--only", nargs="+", help="restrict to these signals")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    built = build_all(set(args.only) if args.only else None)
    if not built:
        raise SystemExit("No signals available. Run: python -m src.train")

    results = [evaluate(s, args.profile) for s in built]

    if len(built) > 1:
        from src.scoring import BlendedSignal
        results.append(evaluate(BlendedSignal(built), args.profile))

    print()
    print_table(results)
    print()
    print(f"corpus: {corpus.size()} films")


if __name__ == "__main__":
    main()
