"""Central configuration: env vars, paths, and tunable model weights.

Everything that used to be read ad-hoc at import time in several modules
lives here, so a missing key fails loudly at startup instead of turning
into a confusing 401 later.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

TMDB_TOKEN   = os.getenv("TMDB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BOT_TOKEN    = os.getenv("BOT_TOKEN")

TMDB_BASE = "https://api.themoviedb.org/3"

PRIMARY_MODEL  = os.getenv("GROQ_PRIMARY_MODEL",  "qwen/qwen3.8-27b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "groq/compound-mini")

CACHE_DB   = DATA_DIR / "movies.db"
MODEL_FILE = DATA_DIR / "model.joblib"
USERS_FILE = DATA_DIR / "users.json"
LEGACY_CACHE_JSON = DATA_DIR / "features_cache.json"

HTTP_TIMEOUT = 15

# How many times each field repeats in a movie's "soup". These are the
# knobs `python -m src.train --tune` optimises against the leave-one-out
# metric, which is why they live in config rather than inline in taste.py.
SOUP_WEIGHTS = {
    "genres":     int(os.getenv("W_GENRES",     3)),
    "director":   int(os.getenv("W_DIRECTOR",   3)),
    "cast":       int(os.getenv("W_CAST",       1)),
    "keywords":   int(os.getenv("W_KEYWORDS",   2)),
    # Fields added in the richer feature schema. Overview terms are
    # deliberately weak: they exist to give sparse rows (documentaries,
    # older films with no billed keywords) enough content that cosine
    # similarity stops over-rewarding their shortness, not to drive
    # matching on their own.
    "overview":   int(os.getenv("W_OVERVIEW",   1)),
    "writers":    int(os.getenv("W_WRITERS",    1)),
    "collection": int(os.getenv("W_COLLECTION", 2)),
    "decade":     int(os.getenv("W_DECADE",     1)),
}

# Ratings at or above this are "liked"; at or below DISLIKE are used as a
# negative signal subtracted from the taste vector.
LIKE_THRESHOLD    = 3.5
DISLIKE_THRESHOLD = 2.0
NEGATIVE_WEIGHT   = 0.3

MAX_LIKED_FILMS = 60

# --- matching quality gates ---------------------------------------------
# A TMDB search result below this score is dropped rather than guessed at.
# Binding a film to the wrong TMDB entry poisons the taste vector with a
# whole set of unrelated genres; dropping it costs only one signal.
MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", 55))

# Thresholds are in standard deviations above the mean score for the
# query, not raw cosine. Raw cosine floors do not survive a change of
# corpus size, feature set, or scoring signal - the same "good match"
# was 0.16 under metadata TF-IDF and 0.66 under embeddings. Expressing
# the bar as "how unusual is this match" is scale-free, so it keeps
# meaning the same thing when the signals underneath change.
MIN_Z = float(os.getenv("MIN_Z", 2.5))

# A list stops where it stops being coherent. The cut is a fraction of
# the best score rather than a fixed distance below it, because how
# peaked the distribution is varies enormously by request: a narrow vibe
# query tops out around 11 sigma while a broad "films like X" tops out
# around 6, and one fixed drop either truncates the first to two picks or
# pads the second out with strangers.
MIN_Z_RETAIN = float(os.getenv("MIN_Z_RETAIN", 0.55))

# When the user names a film ("something like Interstellar"), how much of
# the ranking that film drives versus their taste. The request is about
# the film they asked for; taste is a tiebreak, not the subject.
REFERENCE_WEIGHT = float(os.getenv("REFERENCE_WEIGHT", 0.75))

# A mood is not a title. "Something dark" narrows the field but does not
# pick out a film the way naming one does, so who is asking counts for
# much more. This is a judgement call rather than a measured one - the
# benchmark has no metric for "vague request from a known user".
QUERY_WEIGHT = float(os.getenv("QUERY_WEIGHT", 0.5))

# Rows with fewer than this many distinct tokens are excluded from the
# corpus. Such rows concentrate their whole L2 norm on a handful of terms
# and win cosine comparisons they have no business winning.
MIN_SOUP_TOKENS = int(os.getenv("MIN_SOUP_TOKENS", 8))

# --- reranking -----------------------------------------------------------
# The blend is a bi-encoder: it scores each film against the request
# independently. A second pass lets the LLM look at the request and each
# candidate together and choose the final few - the standard "retrieve
# cheaply, rerank precisely" pattern. RERANK_POOL films clear the floor
# and go to the reranker; it returns the 3 shown. Set RERANK_ENABLED=0 to
# fall straight back to the blend's own top 3.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") != "0"
RERANK_POOL    = int(os.getenv("RERANK_POOL", 15))
RECOMMEND_N    = int(os.getenv("RECOMMEND_N", 3))

# --- user model ----------------------------------------------------------
# A candidate is scored against the K liked films it matches best, not
# against the average of every film watched. Averaging 26 films produces a
# centroid that represents none of them; K=1 is pure nearest-neighbour and
# too jumpy, so a small K smooths without flattening.
TASTE_TOP_K = int(os.getenv("TASTE_TOP_K", 3))

# How much of the taste score comes from the centroid rather than the
# top-K matches. Both were measured, and they win different cases: the
# centroid is better when taste is coherent (it averages out noise), and
# top-K is better when it is not (a centroid of horror and rom-coms
# points at neither). Since real profiles are a mix, so is this.
TASTE_CENTROID_WEIGHT = float(os.getenv("TASTE_CENTROID_WEIGHT", 0.5))

# --- signals -------------------------------------------------------------
# Static distilled embeddings: no torch, ~30MB, numpy-only inference.
EMBED_MODEL = os.getenv("EMBED_MODEL", "minishlab/potion-base-8M")
EMBED_FILE  = DATA_DIR / "embeddings.npz"

# MovieLens gives real behavioural signal without needing our own users.
# links.csv maps its movieId straight onto the TMDB ids we already have.
MOVIELENS_URL     = os.getenv("MOVIELENS_URL",
                              "https://files.grouplens.org/datasets/movielens/ml-32m.zip")
MOVIELENS_DIR     = DATA_DIR / "movielens"
COLLAB_FILE       = DATA_DIR / "collab.npz"
GRAPH_FILE        = DATA_DIR / "graph.npz"
COLLAB_FACTORS    = int(os.getenv("COLLAB_FACTORS", 256))
COLLAB_MIN_RATING = float(os.getenv("COLLAB_MIN_RATING", 3.5))
# Below this many strong ratings the factors are noise, so the signal
# abstains on the film and the blend falls back to content.
COLLAB_MIN_RATINGS_PER_FILM = int(os.getenv("COLLAB_MIN_RATINGS", 20))

# How much each signal counts toward the final score. The two requests
# the bot serves want opposite mixes, so there are two profiles rather
# than one compromise vector (see src.scoring for how they are applied):
#
#   reference ("something like Interstellar") - a film is named, so the
#     precise "shared director / franchise / cast" facts of metadata lead,
#     with semantic adding what-it's-about and collab a little independent
#     taste signal. Grid-searched against src.bench on the 13k corpus:
#     m/s/c = 0.40/0.40/0.20 gave ref MRR 0.149 and hit@10 0.267, against
#     0.133 / 0.250 for the old 0.25/0.35/0.40 - metadata had been weighted
#     *least* in the very context it is strongest, and collab most.
#
#   query ("something dark and mind-bending") - no film named, so only
#     the meaning of the sentence is anything to go on. Semantic dominates;
#     metadata's keyword tokens are a weak assist and collab has no query
#     vector at all. Grid search: m/s = 0.25/0.75 gave qry MRR 0.132,
#     hit@10 0.268, median rank 27, against 0.118 / 0.244 / 43.
SIGNAL_WEIGHTS = {
    "metadata": float(os.getenv("W_METADATA", 0.40)),
    "semantic": float(os.getenv("W_SEMANTIC", 0.40)),
    "collab":   float(os.getenv("W_COLLAB",   0.20)),
    # Built and measured, then benched at zero. TMDB's recommendation
    # graph is a good signal on its own (reference median 60 vs 108 for
    # embeddings) and covers 100% of the corpus where MovieLens covers
    # 46% - but it is roughly twice as correlated with the content
    # signals as MovieLens is (Spearman 0.35 vs 0.19), so it mostly
    # re-derives what metadata already knows. Adding it moved the
    # reference median but cost top-10 hits, and a paired test over the
    # benchmark could not distinguish it from noise (p = 0.39). What
    # earns a signal its place in a blend is independence, not
    # standalone accuracy. Raise this to bring it back.
    "graph":    float(os.getenv("W_GRAPH",    0.0)),
}

# The free-text profile. collab is absent because it cannot encode a
# sentence into its factor space - it only knows films it has rows for.
QUERY_SIGNAL_WEIGHTS = {
    "metadata": float(os.getenv("QW_METADATA", 0.25)),
    "semantic": float(os.getenv("QW_SEMANTIC", 0.75)),
    "collab":   0.0,
    "graph":    0.0,
}

# --- caching -------------------------------------------------------------
# A Letterboxd profile is a scrape of a slow third-party page. Re-using it
# for this long keeps a repeat request instant without going stale.
PROFILE_TTL = int(os.getenv("PROFILE_TTL", 6 * 60 * 60))


def require(*names: str) -> None:
    """Fail fast with a readable message if a required key is missing."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            f"Add them to {PROJECT_ROOT / '.env'}"
        )


DATA_DIR.mkdir(exist_ok=True)
