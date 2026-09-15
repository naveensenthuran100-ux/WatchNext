"""The canonical, ordered list of films every signal aligns to.

Each scoring signal (metadata TF-IDF, dense embeddings, collaborative
factors) produces one vector per corpus row, plus a mask saying which
rows it actually knows about. Sharing one ordering is what makes those
signals blendable: row 400 means the same film to all of them, and a
signal with no data for a film contributes nothing rather than silently
shifting everything by one.
"""
import logging

import numpy as np

from src import cache

log = logging.getLogger(__name__)

_movies = None
_index = None


def movies(refresh: bool = False) -> list[dict]:
    """Every usable cached film, in a stable order (ascending TMDB id)."""
    global _movies, _index
    if _movies is None or refresh:
        rows = [m for m in cache.all_movies() if m.get("genres")]
        _movies = sorted(rows, key=lambda m: m["id"])
        _index = {m["id"]: i for i, m in enumerate(_movies)}
        log.info("corpus: %d films", len(_movies))
    return _movies


def index_by_id() -> dict:
    movies()
    return _index


def size() -> int:
    return len(movies())


def row_of(movie_id: int) -> int | None:
    return index_by_id().get(movie_id)


def rows_of(movie_ids) -> np.ndarray:
    """Corpus rows for the ids we know, dropping the ones we don't."""
    idx = index_by_id()
    return np.array([idx[m] for m in movie_ids if m in idx], dtype=int)


def title_index() -> dict:
    """Lowercased title -> row, for benchmarks that name films."""
    return {m["title"].lower(): i for i, m in enumerate(movies())}
