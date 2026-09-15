"""Content-based taste modelling.

Pipeline: movie features -> a weighted bag-of-words "soup" -> TF-IDF
vector -> a query vector -> cosine similarity against the corpus.

The query vector is the important part. It can be built from two
independent sources:

  * a taste vector - the rating-weighted mean of what a Letterboxd user
    liked, minus a fraction of what they disliked;
  * a reference vector - the films the user actually named ("something
    like Interstellar").

Earlier versions only ever built the first one, so a request that named a
film was answered purely from the requester's viewing history and the
named film was never looked up at all. When both signals exist they are
blended, with the named film dominating: the request is about that film,
and taste is a tiebreak between things already similar to it.
"""
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src import config
from src.cache import cached_features, cached_features_many, cached_search, put_search
from src.letterboxd import clean_title, get_taste_profile
from src.tmdb import search_movie, similar_movie_ids, tmdb_get

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# vectorising
# --------------------------------------------------------------------------

def build_soup(features: dict, weights: dict = None) -> str:
    """Flatten a movie into a repeated-token string.

    Repetition is how a plain TF-IDF vectorizer gets told that a genre
    match matters more than a shared bit-part actor.
    """
    weights = weights or config.SOUP_WEIGHTS

    def clean(text: str) -> str:
        return str(text or "").lower().replace(" ", "")

    def w(name: str, default: int = 1) -> int:
        return int(weights.get(name, default))

    genres    = [clean(g) for g in features.get("genres", [])]
    keywords  = [clean(k) for k in features.get("keywords", [])]
    cast      = [clean(c) for c in features.get("cast", [])]
    writers   = [clean(x) for x in features.get("writers", [])]
    overview  = [clean(t) for t in features.get("overview_terms", [])]
    directors = [clean(d) for d in features.get("directors", [])] or \
                ([clean(features.get("director"))] if features.get("director") else [])
    collection = clean(features.get("collection", ""))
    decade     = clean(features.get("decade", ""))

    tokens = (
        genres * w("genres", 3)
        + directors * w("director", 3)
        + cast * w("cast", 1)
        + keywords * w("keywords", 2)
        + writers * w("writers", 1)
        + overview * w("overview", 1)
        + ([collection] * w("collection", 2) if collection else [])
        + ([decade] * w("decade", 1) if decade else [])
    )
    return " ".join(t for t in tokens if t)


def soup_tokens(features: dict, weights: dict = None) -> int:
    """Distinct tokens a movie contributes - its vector's information content."""
    return len(set(build_soup(features, weights).split()))


def fit_vectorizer(all_soups: list) -> tuple:
    """Fit TF-IDF over the corpus.

    min_df=2 drops terms appearing in a single movie. Those terms can
    never match anything else, but they do absorb L2 norm, which was
    part of why sparse rows scored so strangely.
    """
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(all_soups)
    return vectorizer, matrix


# --------------------------------------------------------------------------
# query vectors
# --------------------------------------------------------------------------

def _rating_weight(rating: float) -> float:
    """Turn a 0-5 star rating into a positive weight with real spread.

    Using the raw rating would make a 3.5 and a 5.0 nearly equivalent
    (weights 3.5 vs 5.0); this maps them to 1.0 vs 2.5.
    """
    return max(rating - 2.5, 0.1)


def build_taste_vector(liked_vectors, star_ratings: list, disliked_vectors=None) -> np.ndarray:
    """Rating-weighted centroid of liked films, minus a disliked centroid."""
    dense = liked_vectors.toarray() if hasattr(liked_vectors, "toarray") else np.asarray(liked_vectors)
    if dense.shape[0] == 0:
        raise ValueError("cannot build a taste vector from zero films")

    weights = np.array([_rating_weight(r) for r in star_ratings], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(star_ratings))

    taste = np.average(dense, axis=0, weights=weights)

    if disliked_vectors is not None and disliked_vectors.shape[0] > 0:
        neg = disliked_vectors.toarray() if hasattr(disliked_vectors, "toarray") else np.asarray(disliked_vectors)
        taste = taste - config.NEGATIVE_WEIGHT * neg.mean(axis=0)
        taste = np.clip(taste, 0, None)

    return taste.reshape(1, -1)


def build_reference_vector(vectorizer, reference_features: list, weights: dict,
                           extra_keywords: list = None) -> np.ndarray | None:
    """A query vector for the films (and terms) the user actually named.

    Named films carry the bulk of the meaning; loose keywords the model
    pulled out of the message ("space", "physics") are folded in at a
    lower weight so they sharpen the match without overriding the film.
    """
    parts, part_weights = [], []

    for features in reference_features or []:
        parts.append(vectorizer.transform([build_soup(features, weights)]).toarray()[0])
        part_weights.append(1.0)

    terms = [str(k).lower().replace(" ", "") for k in (extra_keywords or []) if k]
    if terms:
        keyword_vector = vectorizer.transform([" ".join(terms)]).toarray()[0]
        if keyword_vector.any():
            parts.append(keyword_vector)
            part_weights.append(0.35 if reference_features else 1.0)

    if not parts:
        return None

    stacked = np.average(np.vstack(parts), axis=0, weights=part_weights)
    return stacked.reshape(1, -1)


def _cosine(query, matrix) -> np.ndarray:
    if query is None:
        return None
    return cosine_similarity(normalize(query), matrix)[0]


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def rank_candidates(taste_vector, candidate_vectors, candidates: list) -> list:
    """Score and sort candidates.

    Returns new dicts - the originals may be cache-owned, and annotating
    them in place used to leak a "score" field into the stored features.
    """
    if not candidates:
        return []
    sims = cosine_similarity(taste_vector, candidate_vectors)[0]
    scored = [dict(c, score=float(s)) for c, s in zip(candidates, sims)]
    return sorted(scored, key=lambda c: c["score"], reverse=True)


def explain_match(candidate: dict, reference_features: list = None,
                  liked_features: list = None, query_keywords: list = None,
                  query_genres: list = None) -> dict:
    """Concrete overlaps between a pick and what it was matched against.

    This exists so the write-up stage has facts to work from. Previously
    the model was handed the raw request and a list of films chosen by an
    unrelated mechanism and asked why they fit, so it invented a reason -
    which is how "Interstellar" came back with Little Women described as
    the same sweeping emotional journey.
    """
    sources = list(reference_features or []) + list(liked_features or [])[:8]

    # A request with no named film and no profile still has to justify
    # itself, or every vibe recommendation is written up as having "no
    # computed overlap" - technically honest, and useless to read.
    if not sources:
        return _explain_against_query(candidate, query_keywords, query_genres)

    def pool(field: str) -> set:
        out = set()
        for f in sources:
            out.update(x for x in f.get(field, []) if x)
        return out

    directors = {f.get("director") for f in sources if f.get("director")}
    reason = {
        "genres":   [g for g in candidate.get("genres", []) if g in pool("genres")][:3],
        "keywords": [k for k in candidate.get("keywords", []) if k in pool("keywords")][:4],
        "cast":     [c for c in candidate.get("cast", []) if c in pool("cast")][:2],
    }
    if candidate.get("director") and candidate["director"] in directors:
        reason["director"] = candidate["director"]
    return {k: v for k, v in reason.items() if v}


# --------------------------------------------------------------------------
# resolving titles -> features
# --------------------------------------------------------------------------

def _explain_against_query(candidate: dict, query_keywords: list = None,
                           query_genres: list = None) -> dict:
    """Overlaps between a pick and the request itself."""
    wanted_genres = {g.lower() for g in (query_genres or [])}
    genres = [g for g in candidate.get("genres", []) if g.lower() in wanted_genres]

    terms = [str(k).lower() for k in (query_keywords or []) if k]
    keywords = []
    for keyword in candidate.get("keywords", []):
        low = str(keyword).lower()
        if any(t in low or low in t for t in terms):
            keywords.append(keyword)

    reason = {"genres": genres[:3], "keywords": keywords[:4]}
    return {k: v for k, v in reason.items() if v}


def resolve_one(title: str, year: str = None) -> dict | None:
    """Title (+ optional year) -> TMDB feature dict, via the search cache."""
    title = clean_title(title)
    if not title:
        return None

    was_cached, movie_id = cached_search(title, year)
    if not was_cached:
        try:
            result = search_movie(title, year)
        except Exception as exc:
            log.warning("TMDB search failed for %r: %s", title, exc)
            return None
        movie_id = result["id"] if result else None
        put_search(title, year, movie_id)

    if movie_id is None:
        return None
    try:
        return cached_features(movie_id)
    except Exception as exc:
        log.warning("could not fetch features for %r: %s", title, exc)
        return None


def resolve_films(films: list) -> tuple[list, list]:
    """Letterboxd entries -> (TMDB feature dicts, ratings), year-disambiguated."""
    features, ratings = [], []
    for film in films:
        resolved = resolve_one(film["title"], film.get("year"))
        if resolved is None:
            continue
        features.append(resolved)
        ratings.append(film.get("rating") or 4.0)
    return features, ratings


def resolve_titles(titles: list) -> list:
    """Film names pulled out of a chat message -> feature dicts."""
    out = []
    for title in titles or []:
        resolved = resolve_one(str(title))
        if resolved and resolved["id"] not in {f["id"] for f in out}:
            out.append(resolved)
    return out


# --------------------------------------------------------------------------
# candidate pools
# --------------------------------------------------------------------------

def get_candidate_pool(liked_features: list, limit: int = 300,
                       reference_features: list = None) -> list:
    """Live TMDB fallback used when no trained model is available.

    Collects ids first and truncates before fetching, so a limit of 300
    means 300 enrichment calls rather than 300 after fetching ~400.
    """
    seeds = list(reference_features or []) + list(liked_features or [])
    if not seeds:
        return []

    genre_data = tmdb_get("/genre/movie/list")
    genre_map  = {g["name"]: g["id"] for g in genre_data.get("genres", [])}

    seed_genres = []
    for f in seeds:
        for g in f.get("genres", []):
            if g not in seed_genres:
                seed_genres.append(g)
    genre_ids = [str(genre_map[g]) for g in seed_genres if g in genre_map]

    ids = []
    # Neighbours of the named films first - they are the most on-target
    # candidates we can get, and they cost one request each.
    for f in (reference_features or [])[:3]:
        ids.extend(similar_movie_ids(f["id"]))

    if genre_ids:
        for page in range(1, 6):
            data = tmdb_get("/discover/movie", params={
                "with_genres":    ",".join(genre_ids),
                "sort_by":        "popularity.desc",
                "vote_count.gte": 100,
                "page":           page,
            })
            ids.extend(m["id"] for m in data.get("results", []))

    for f in (liked_features or [])[:3]:
        ids.extend(similar_movie_ids(f["id"], limit=20))

    seed_ids = {f["id"] for f in seeds}
    unique = [i for i in dict.fromkeys(ids) if i not in seed_ids][:limit]
    return cached_features_many(unique)


def _watched_key(title: str, year) -> tuple:
    return (str(title or "").lower().strip(), str(year) if year else None)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def _standardise(scores, covered) -> np.ndarray | None:
    """Express scores as standard deviations above the mean.

    Every threshold downstream is in these units, so that the bar for
    "this is a real match" survives swapping the signals underneath it.
    """
    if scores is None:
        return None
    values = scores[covered]
    if values.size < 2 or values.std() == 0:
        return None
    out = np.full(scores.shape, -np.inf)
    out[covered] = (values - values.mean()) / values.std()
    return out


def _reference_scores(signal, reference_features):
    """Similarity to the films the user named, averaged over them."""
    per_film = []
    for features in reference_features or []:
        sims = signal.similarity(signal.encode_movie(features), context="reference")
        if sims is not None:
            per_film.append(sims)
    if not per_film:
        return None
    return np.mean(np.vstack(per_film), axis=0)


def recommend(profile: dict = None, reference_features: list = None,
              keywords: list = None, query_text: str = None,
              genre_filter: list = None, signal=None, top_n: int = 8,
              min_z: float = None) -> list:
    """Rank the corpus against whatever the request actually gives us.

    Three inputs, any of which may be absent:

      reference_features  films the user named ("like Interstellar")
      query_text          what they typed ("dark and mind-bending")
      profile             who they are (their Letterboxd history)

    What they asked for leads; who they are breaks ties. Doing it the
    other way round is what produced the original failure, where a
    request naming Interstellar was answered entirely from the
    requester's viewing history and the named film was never looked up.
    """
    from src import scoring

    min_z = config.MIN_Z if min_z is None else min_z
    reference_features = reference_features or []

    signal = signal or scoring.load_default()
    if signal is None:
        log.warning("no scoring signal available - run: python -m src.train")
        return []

    candidates = signal.movies

    liked_features, liked_ratings, disliked_features = [], [], []
    if profile:
        liked_features, liked_ratings = resolve_films(profile["liked"])
        disliked_features, _ = resolve_films(profile["disliked"][:15])

    if not liked_features and not reference_features and not keywords and not query_text:
        return []

    # --- what they asked for ---------------------------------------------
    intent = _reference_scores(signal, reference_features)
    intent_weight = config.REFERENCE_WEIGHT
    if intent is None and (query_text or keywords):
        intent = signal.similarity(signal.encode_query(query_text or "", keywords))
        intent_weight = config.QUERY_WEIGHT

    # --- who they are ------------------------------------------------------
    taste = None
    if liked_features:
        taste = signal.profile_similarity(
            [signal.encode_movie(f) for f in liked_features], liked_ratings)

    if intent is not None and taste is not None:
        scores = intent_weight * intent + (1.0 - intent_weight) * taste
    else:
        scores = intent if intent is not None else taste
    if scores is None:
        return []

    scores = _standardise(np.asarray(scores, dtype=np.float64), signal.covered)
    if scores is None:
        return []

    # --- filters ------------------------------------------------------------
    mask = np.ones(len(candidates), dtype=bool)

    # A documentary and a feature are different things to watch, however
    # much vocabulary they share. Asking for films like Interstellar was
    # returning "Interstellar: Nolan's Odyssey", the making-of, third.
    if reference_features:
        ref_is_doc = any("Documentary" in f.get("genres", [])
                         for f in reference_features)
        mode = np.array([("Documentary" in m.get("genres", [])) == ref_is_doc
                         and "TV Movie" not in m.get("genres", [])
                         for m in candidates])
        if mode.sum() >= 10:
            mask &= mode

    # Genre is a far worse constraint than a named film, so it is only
    # applied when no film was named. Filtering "like Interstellar" down
    # to Drama throws away good matches to no purpose.
    if genre_filter and not reference_features:
        wanted = {g.lower() for g in genre_filter}
        genre = np.array([any(g.lower() in wanted for g in m.get("genres", []))
                          for m in candidates])
        if genre.sum() >= 10:
            mask &= genre

    seen_ids = ({f["id"] for f in liked_features}
                | {f["id"] for f in disliked_features}
                | {f["id"] for f in reference_features})
    seen_keys = (profile or {}).get("watched_keys", set())
    for i, m in enumerate(candidates):
        if m["id"] in seen_ids or _watched_key(m.get("title"), m.get("year")) in seen_keys:
            mask[i] = False

    scores = np.where(mask, scores, -np.inf)

    # --- pick ---------------------------------------------------------------
    order = np.argsort(scores)[::-1]
    best = scores[order[0]]
    if not np.isfinite(best) or best < min_z:
        return []

    floor = max(min_z, best * config.MIN_Z_RETAIN)

    # With only a handful of slots, two sequels from the same franchise
    # eating a third of the list is a worse list than swapping one for
    # the next-best thing outside it - so each collection gets one seat
    # on the first pass, and only fills a second seat if nothing else
    # cleared the floor.
    picks, seen_collections, deferred = [], set(), []
    for i in order:
        if scores[i] < floor:
            break
        film = dict(candidates[i], score=float(scores[i]))
        film["why"] = explain_match(film, reference_features, liked_features,
                                    query_keywords=keywords,
                                    query_genres=genre_filter)

        # If the user named a film, a pick we cannot connect to it in any
        # concrete way is exactly the old failure mode - presented as
        # "like Interstellar" on the strength of nothing at all.
        if reference_features and not film["why"]:
            continue

        collection = film.get("collection")
        if collection and collection in seen_collections:
            deferred.append(film)
            continue

        if collection:
            seen_collections.add(collection)
        picks.append(film)
        if len(picks) >= top_n:
            return picks

    for film in deferred:
        picks.append(film)
        if len(picks) >= top_n:
            break

    return picks


def recommend_for_profile(profile: dict, signal=None, top_n: int = 10,
                          genre_filter: list = None) -> list:
    """Rank the corpus for a Letterboxd profile, excluding what they've seen."""
    return recommend(profile=profile, genre_filter=genre_filter,
                     signal=signal, top_n=top_n)


def recommend_for_letterboxd_user(username: str, top_n: int = 10) -> list:
    profile = get_taste_profile(username)
    if not profile:
        return []
    return recommend_for_profile(profile, top_n=top_n)
