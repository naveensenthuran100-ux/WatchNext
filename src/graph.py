"""Collaborative signal from TMDB's own recommendation graph.

    python -m src.graph          # fetch the graph, build vectors, save

MovieLens gives real behavioural signal but only covers 46% of the
corpus - it is a 2015 dataset, so anything recent is invisible to it, and
that is most of what people ask about.

TMDB's /recommendations endpoint is also behaviour-derived (it comes from
what its users watch and rate together, unlike /similar, which is built
from keywords and genres). It covers every film in the corpus, including
last month's releases. So it gives the same *kind* of signal as
MovieLens - films connected by what audiences actually pair them with,
rather than by shared metadata - without the coverage hole.

Each film is represented by the set of films TMDB associates with it.
Two films are similar when those sets overlap, which is second-order:
they need not be linked to each other, only to the same neighbourhood.
"""
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import normalize

from src import cache, config, corpus, signals
from src.tmdb import tmdb_get

log = logging.getLogger(__name__)


class GraphSignal(signals.Signal):
    """Films are close when TMDB's audiences pair them with the same films."""

    name = "graph"

    def encode_query(self, text: str, keywords=None):
        return None                 # a co-view graph says nothing about prose

    def encode_movie(self, features: dict):
        existing = super().encode_movie(features)
        if existing is not None:
            return existing

        # A film outside the corpus still has neighbours; fetch them so a
        # reference film we have never indexed can still be matched.
        movie_id = features.get("id")
        if movie_id is None:
            return None
        neighbours = cache.get_recommendations(movie_id)
        if neighbours is None:
            try:
                neighbours = fetch_recommendations(movie_id)
                cache.put_recommendations(movie_id, neighbours)
            except Exception as exc:
                log.warning("could not fetch recommendations for %s: %s",
                            movie_id, exc)
                return None

        rows = corpus.rows_of(neighbours)
        if rows.size == 0:
            return None
        vector = np.zeros(self.vectors.shape[1], dtype=np.float32)
        vector[rows] = 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else None


def fetch_recommendations(movie_id: int, pages: int = 2) -> list:
    """TMDB's recommended films for one film, most relevant first."""
    ids = []
    for page in range(1, pages + 1):
        data = tmdb_get(f"/movie/{movie_id}/recommendations",
                        params={"page": page})
        results = data.get("results", [])
        ids.extend(m["id"] for m in results)
        if len(results) < 20:
            break
    return list(dict.fromkeys(ids))


def fetch_graph(workers: int = 12, refresh: bool = False) -> int:
    """Fetch recommendations for every corpus film, caching as it goes."""
    films = corpus.movies()
    known = cache.all_recommendations()
    todo = [m["id"] for m in films if refresh or m["id"] not in known]

    log.info("recommendation graph: %d films, %d already cached, %d to fetch",
             len(films), len(films) - len(todo), len(todo))
    if not todo:
        return 0

    done, started = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_recommendations, i): i for i in todo}
        for future in as_completed(futures):
            movie_id = futures[future]
            try:
                cache.put_recommendations(movie_id, future.result())
            except Exception as exc:
                log.warning("film %s failed: %s", movie_id, exc)
            done += 1
            if done % 500 == 0:
                log.info("  fetched %d/%d (%.1f/s)",
                         done, len(todo), done / (time.time() - started))
    return done


def build_adjacency() -> sp.csr_matrix:
    """Symmetric film-to-film adjacency over the corpus."""
    films = corpus.movies()
    index = corpus.index_by_id()
    edges = cache.all_recommendations()

    rows, cols, vals = [], [], []
    for movie_id, neighbours in edges.items():
        i = index.get(int(movie_id))
        if i is None:
            continue
        # Earlier entries are the stronger associations, so they count for
        # more; a flat weighting lets the 40th recommendation speak as
        # loudly as the 1st.
        for rank, neighbour in enumerate(neighbours):
            j = index.get(int(neighbour))
            if j is None or j == i:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(1.0 / np.log2(rank + 2))

    n = len(films)
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)

    matrix = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n))

    # The relation is mutually informative but TMDB only reports it one
    # way for many pairs, so it is symmetrised before use.
    matrix = matrix.maximum(matrix.T)

    # A film counts as its own neighbour, so that a direct edge between
    # two films contributes even when they share no other neighbours.
    matrix = matrix + sp.identity(n, format="csr", dtype=np.float32)
    return matrix.tocsr()


def build(factors: int = 0) -> GraphSignal:
    """Build the signal. factors>0 reduces the graph to dense components."""
    adjacency = build_adjacency()
    degree = np.asarray((adjacency != 0).sum(axis=1)).reshape(-1)
    covered = degree > 1                # more than just its own self-loop

    if factors:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=factors, random_state=0)
        vectors = normalize(np.asarray(svd.fit_transform(adjacency),
                                       dtype=np.float32))
        vectors[~covered] = 0.0
        log.info("graph reduced to %d factors, explained variance %.2f",
                 factors, float(svd.explained_variance_ratio_.sum()))
    else:
        vectors = normalize(adjacency).tocsr()
        if (~covered).any():
            vectors = (sp.diags(covered.astype(np.float32)) @ vectors).tocsr()

    log.info("graph: %d films covered (%.1f%%), mean degree %.1f",
             int(covered.sum()), 100.0 * covered.mean(),
             float(degree[covered].mean()) if covered.any() else 0.0)
    return GraphSignal(vectors, covered)


def save(signal: GraphSignal, path=None) -> None:
    path = path or config.GRAPH_FILE
    ids = np.array([m["id"] for m in corpus.movies()], dtype=np.int64)
    matrix = signal.vectors
    if sp.issparse(matrix):
        matrix = matrix.tocoo()
        np.savez_compressed(
            path, sparse=True, data=matrix.data, row=matrix.row, col=matrix.col,
            shape=np.array(matrix.shape), covered=signal.covered, ids=ids)
    else:
        np.savez_compressed(path, sparse=False, vectors=matrix,
                            covered=signal.covered, ids=ids)
    log.info("saved graph signal to %s", path)


def load_graph(path=None) -> GraphSignal | None:
    """Load the graph, realigned onto the current corpus ordering."""
    path = path or config.GRAPH_FILE
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return None

    saved_ids = data["ids"]
    if bool(data["sparse"]):
        saved = sp.csr_matrix((data["data"], (data["row"], data["col"])),
                              shape=tuple(data["shape"]))
    else:
        saved = data["vectors"]
    saved_covered = data["covered"]

    films = corpus.movies()
    current = [m["id"] for m in films]
    if len(current) == len(saved_ids) and (saved_ids == np.array(current)).all():
        return GraphSignal(saved, saved_covered)

    # Corpus changed since the graph was built: rebuild rather than
    # silently scoring films against another corpus's row numbers.
    log.warning("graph is stale (corpus changed) - run: python -m src.graph")
    return None


def main():
    parser = argparse.ArgumentParser(description="Build the TMDB graph signal")
    parser.add_argument("--factors", type=int, default=0,
                        help="reduce to N dense factors (0 = keep sparse)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch edges even if cached")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.require("TMDB_TOKEN")

    fetch_graph(workers=args.workers, refresh=args.refresh)
    signal = build(factors=args.factors)
    save(signal)

    print("\nTMDB recommendation graph built")
    print(f"  films covered: {int(signal.covered.sum())} / {corpus.size()} "
          f"({signal.coverage:.1%})")
    print(f"  saved to:      {config.GRAPH_FILE}")


if __name__ == "__main__":
    main()
