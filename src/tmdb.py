"""TMDB API client.

Adds the three things the original was missing: a timeout on every call,
an error check (so a 401 stops being an empty result list), and automatic
retry/backoff on 429 so building a candidate pool doesn't die halfway.

Two later changes matter for quality:

1. `get_movie_features` uses append_to_response, so enriching one movie is
   a single request instead of three. That is what makes a 5000-movie
   corpus practical rather than a 15000-request afternoon.
2. `search_movie` scores candidates instead of trusting results[0]. The
   old behaviour bound the TV series "Baby Reindeer" to a documentary
   called "A Baby Reindeer's First Christmas", and two mismatches like
   that were enough to make every recommendation a documentary.
"""
import logging
import re
import unicodedata

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config

log = logging.getLogger(__name__)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=20))
    return session


_session = _build_session()


def tmdb_get(path: str, params: dict = None) -> dict:
    config.require("TMDB_TOKEN")
    url = f"{config.TMDB_BASE}{path}"
    response = _session.get(
        url,
        headers={"Authorization": f"Bearer {config.TMDB_TOKEN}"},
        params=params,
        timeout=config.HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise_title(title: str) -> str:
    """Casefold, strip accents and punctuation: 'WALL·E' -> 'walle'."""
    decomposed = unicodedata.normalize("NFKD", str(title or ""))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _PUNCT.sub("", stripped.lower()).strip()


def _score_candidate(candidate: dict, want_title: str, want_year: str | None) -> float:
    """How well a TMDB search result matches what we asked for.

    Title agreement dominates, year is a strong secondary signal, and
    popularity only ever breaks ties between equally good matches.
    """
    got = normalise_title(candidate.get("title") or "")
    alt = normalise_title(candidate.get("original_title") or "")

    def overlap(a: str, b: str) -> float:
        """How much of the longer title the shorter one accounts for.

        Without this, any short title matches any longer one containing
        it: the TV series "Adolescence" bound to a documentary called
        "The Real Adolescence: Our Killer Kids" purely on containment,
        and dragged Documentary into the taste vector behind it.
        """
        return min(len(a), len(b)) / max(len(a), len(b), 1)

    if want_title in (got, alt):
        score = 100.0
    elif got.startswith(want_title) or want_title.startswith(got):
        score = 60.0 * overlap(want_title, got)
    elif want_title in got or got in want_title:
        score = 30.0 * overlap(want_title, got)
    else:
        return -1.0                     # unrelated title: never accept

    if want_year:
        got_year = (candidate.get("release_date") or "")[:4]
        if got_year == want_year:
            score += 40.0
        elif got_year and abs(int(got_year) - int(want_year)) <= 1:
            score += 20.0               # Letterboxd and TMDB disagree by a year
        elif got_year:
            score -= 15.0

    votes = candidate.get("vote_count") or 0
    score += min(votes, 5000) / 1000.0                      # up to +5
    score += min(candidate.get("popularity") or 0.0, 100.0) / 50.0   # up to +2
    return score


def search_movie(title: str, year: str = None) -> dict | None:
    """Look up a film, preferring an exact title+year match.

    Returns None rather than a bad guess. A dropped film costs us one
    signal; a wrong film poisons the taste vector with a whole set of
    unrelated genres, and that is much more expensive.
    """
    want = normalise_title(title)
    if not want:
        return None

    seen, candidates = set(), []
    queries = [{"query": title, "primary_release_year": year}] if year else []
    queries.append({"query": title})

    for params in queries:
        try:
            results = tmdb_get("/search/movie", params=params).get("results", [])
        except Exception as exc:
            log.warning("TMDB search failed for %r: %s", title, exc)
            continue
        for candidate in results[:10]:
            if candidate["id"] not in seen:
                seen.add(candidate["id"])
                candidates.append(candidate)
        if candidates and params.get("primary_release_year"):
            break                       # a year-qualified hit is good enough

    if not candidates:
        return None

    best = max(candidates, key=lambda c: _score_candidate(c, want, year))
    if _score_candidate(best, want, year) < config.MIN_SEARCH_SCORE:
        log.info("no confident TMDB match for %r (%s) - dropping", title, year)
        return None
    return best


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

# Bumped whenever the shape of a feature dict changes, so cached rows
# written by an older version are refetched instead of silently mixing
# thin and rich vectors in the same corpus.
FEATURE_VERSION = 2

# Bumped whenever `search_movie`'s matching logic changes. Without this,
# improving the matcher fixes nothing for existing users: the wrong
# title -> id bindings the old logic wrote stay cached forever, and the
# taste vectors built from them stay just as poisoned.
SEARCH_VERSION = 2

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "his", "her", "its",
    "but", "who", "she", "him", "they", "them", "their", "has", "have", "was",
    "were", "are", "been", "when", "where", "what", "will", "would", "into",
    "out", "off", "over", "after", "before", "then", "than", "not", "all",
    "one", "two", "new", "now", "get", "gets", "must", "can", "back", "own",
    "way", "him", "himself", "herself", "while", "about", "which", "there",
    "story", "film", "movie", "life", "man", "woman", "young", "old", "world",
    "years", "year", "time", "day", "night", "find", "finds", "take", "takes",
    "make", "makes", "come", "comes", "help", "set", "him.", "only", "him,",
}

_WORD = re.compile(r"[a-z][a-z'-]{2,}")


def overview_terms(overview: str, limit: int = 12) -> list[str]:
    """Content words from the synopsis, in order of first appearance.

    Documentaries and older films often carry almost no keywords or
    billed cast in TMDB, which left their vectors with five or six
    non-zero terms. Cosine similarity badly over-rewards such short
    rows, so every sparse documentary floated to the top of any ranking
    that had even a trace of documentary in it. Synopsis terms give
    those rows real content to match on.
    """
    words, seen = [], set()
    for match in _WORD.finditer(str(overview or "").lower()):
        word = match.group(0).strip("'-")
        if len(word) < 4 or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= limit:
            break
    return words


def _decade(year: str) -> str:
    return f"{year[:3]}0s" if year and year.isdigit() and len(year) == 4 else ""


def get_movie_features(movie_id: int) -> dict:
    """Everything we model a movie on, in one HTTP request."""
    data = tmdb_get(
        f"/movie/{movie_id}",
        params={"append_to_response": "keywords,credits"},
    )

    credits = data.get("credits") or {}
    crew = credits.get("crew") or []

    directors = [p["name"] for p in crew if p.get("job") == "Director"]
    writers = [p["name"] for p in crew
               if p.get("job") in ("Screenplay", "Writer", "Story")]
    composers = [p["name"] for p in crew if p.get("job") == "Original Music Composer"]

    year = (data.get("release_date") or "")[:4]
    collection = (data.get("belongs_to_collection") or {}).get("name") or ""

    return {
        "_v":           FEATURE_VERSION,
        "id":           movie_id,
        "title":        data.get("title", ""),
        "year":         year,
        "decade":       _decade(year),
        "genres":       [g["name"] for g in data.get("genres") or []],
        "keywords":     [k["name"] for k in (data.get("keywords") or {}).get("keywords") or []][:20],
        "cast":         [c["name"] for c in (credits.get("cast") or [])[:8]],
        "director":     directors[0] if directors else "",
        "directors":    directors[:2],
        "writers":      writers[:3],
        "composer":     composers[0] if composers else "",
        "collection":   collection,
        "companies":    [c["name"] for c in data.get("production_companies") or []][:2],
        "language":     data.get("original_language") or "",
        "runtime":      data.get("runtime") or 0,
        "tagline":      data.get("tagline") or "",
        "overview":     data.get("overview") or "",
        "overview_terms": overview_terms(data.get("overview") or ""),
        "vote_average": data.get("vote_average") or 0.0,
        "vote_count":   data.get("vote_count") or 0,
        "popularity":   data.get("popularity") or 0.0,
    }


def similar_movie_ids(movie_id: int, limit: int = 40) -> list[int]:
    """TMDB's own neighbours for a film, used to widen the candidate pool."""
    ids = []
    for path in (f"/movie/{movie_id}/recommendations", f"/movie/{movie_id}/similar"):
        try:
            ids.extend(m["id"] for m in tmdb_get(path).get("results", []))
        except Exception as exc:
            log.warning("could not fetch %s: %s", path, exc)
    return list(dict.fromkeys(ids))[:limit]
