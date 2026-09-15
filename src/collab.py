"""Collaborative signal from MovieLens.

    python -m src.collab            # download, build item factors, save

Content-based matching has a hard ceiling: it can only see what somebody
wrote down. Tone, pacing, humour, whether a film is comforting or
punishing - none of that is in a genre list, so no amount of metadata
weighting will make "films that feel like Interstellar" work properly.

What captures that is behaviour: the same people liking both. This bot
has no user base to learn from, but it does not need one. MovieLens is a
free dataset of tens of millions of real ratings, and it ships links.csv
mapping its own movieId straight onto the TMDB ids the corpus already
uses - so the behavioural signal drops onto the existing films with no
identity resolution to get wrong.

Method: build a sparse (corpus film x user) matrix of who rated what
highly, damp the popularity bias, then reduce to dense factors with a
truncated SVD. Two films end up close when the same people liked both.
"""
import argparse
import io
import logging
import re
import time
import urllib.request
import zipfile

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src import config, corpus, signals

log = logging.getLogger(__name__)


class CollabSignal(signals.Signal):
    """Films are close when the same people liked both.

    Coverage is partial by nature - MovieLens has never heard of a film
    released last month - so this signal abstains on those rather than
    guessing, and the blend leans on content for them instead.
    """

    name = "collab"

    def encode_query(self, text: str, keywords=None):
        return None                 # behaviour has nothing to say about prose


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

# The official host, files.grouplens.org, is the preferred source. Its
# TLS certificate expired on 2026-08-28, so a mirror is used when the
# official download cannot be verified. Verification is never disabled -
# a broken certificate is a reason to fetch from somewhere else, not a
# reason to stop checking.
MIRROR_URL = ("https://huggingface.co/datasets/auxten/movielens-20m/"
              "resolve/main/movielens-20m.db.zip")


def _unpack(payload: bytes, dest) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest)


SOURCE_FILE = "source.json"


def read_provenance(dest=None) -> dict:
    """Which dataset is on disk, and where it came from."""
    dest = dest or config.MOVIELENS_DIR
    path = dest / SOURCE_FILE
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_provenance(dest, source: str, url: str) -> None:
    import json
    (dest / SOURCE_FILE).write_text(json.dumps(
        {"source": source, "url": url, "fetched_at": time.time()}, indent=2))


def official_available(url: str = None, timeout: int = 20) -> tuple[bool, str]:
    """Is the official GroupLens download usable right now?

    Kept as a first-class check because the reason it is unusable is
    temporary and outside our control: as of 2026-09 the certificate on
    files.grouplens.org expired on 2026-08-28. Verification is never
    disabled to work around that - when it is fixed, this starts
    returning True and `--upgrade` switches over.
    """
    url = url or config.MOVIELENS_URL
    try:
        import urllib.error
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "watchnext-bot"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200, f"HTTP {response.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def find_dataset(dest=None):
    """Locate an already-downloaded MovieLens, in either supported layout.

    Returns ("csv", directory) for the official release, or
    ("sqlite", file) for the mirrored database.
    """
    dest = dest or config.MOVIELENS_DIR
    if not dest.exists():
        return None

    csv_dirs = list(dest.glob("*/ratings.csv"))
    if csv_dirs:
        return "csv", csv_dirs[0].parent

    databases = list(dest.glob("*.db"))
    if databases:
        return "sqlite", databases[0]
    return None


def download(url: str = None, dest=None, upgrade: bool = False):
    """Fetch MovieLens if it isn't already on disk.

    `upgrade` re-attempts the official source even when mirrored data is
    already present, so a fixed certificate can be picked up without
    deleting anything by hand.
    """
    dest = dest or config.MOVIELENS_DIR
    dest.mkdir(parents=True, exist_ok=True)

    found = find_dataset(dest)
    provenance = read_provenance(dest)

    if found and not upgrade:
        log.info("using existing MovieLens data at %s (source: %s)",
                 found[1], provenance.get("source", "unknown"))
        return found

    if found and provenance.get("source") == "official":
        log.info("already on the official dataset - nothing to upgrade")
        return found

    if upgrade:
        ok, detail = official_available(url)
        if not ok:
            log.warning("official source still unavailable (%s) - keeping "
                        "the current dataset", detail)
            if found:
                return found

    for label, source in (("official", url or config.MOVIELENS_URL),
                          ("mirror", MIRROR_URL)):
        log.info("downloading MovieLens from the %s source: %s", label, source)
        try:
            started = time.time()
            request = urllib.request.Request(
                source, headers={"User-Agent": "watchnext-bot"})
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = response.read()
            log.info("downloaded %.0f MB in %.0fs",
                     len(payload) / 1e6, time.time() - started)
            _unpack(payload, dest)
            found = find_dataset(dest)
            if found:
                write_provenance(dest, label, source)
                return found
        except Exception as exc:
            log.warning("%s source failed: %s", label, exc)

    raise RuntimeError(
        "Could not download MovieLens from either source. Download it "
        "manually from https://grouplens.org/datasets/movielens/ and "
        f"unzip it into {dest}")


# --------------------------------------------------------------------------
# reading either layout
# --------------------------------------------------------------------------

_RATING_COLUMNS = ["userId", "movieId", "rating"]
_RATING_DTYPES = {"userId": np.int64, "movieId": np.int64, "rating": np.float32}


def load_movie_titles(source) -> "pd.DataFrame":
    import pandas as pd

    kind, path = source
    if kind == "csv":
        return pd.read_csv(path / "movies.csv", usecols=["movieId", "title"])

    import sqlite3
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query("SELECT movieId, title FROM movies", conn)


def iter_ratings(source, chunk: int):
    """Yield rating chunks, whichever layout the data is in."""
    import pandas as pd

    kind, path = source
    if kind == "csv":
        yield from pd.read_csv(path / "ratings.csv", usecols=_RATING_COLUMNS,
                               dtype=_RATING_DTYPES, chunksize=chunk)
        return

    # The mirror splits ratings into train/test by user for its own
    # benchmark; for building item factors we want all of them.
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        sources = [t for t in ("ratings", "ratings_train", "ratings_test")
                   if t in tables]
        for table in sources:
            query = f"SELECT userId, movieId, rating FROM {table}"
            yield from pd.read_sql_query(query, conn, chunksize=chunk)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# mapping MovieLens films onto the corpus
# --------------------------------------------------------------------------

# MovieLens stores titles with the article moved to the end:
# "Godfather, The (1972)". Left alone, every such film fails to match.
_TRAILING_ARTICLE = re.compile(r"^(.*),\s+(The|A|An|La|Le|Les|El|Il)$",
                               re.IGNORECASE)
_TITLE_YEAR = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def _split_title_year(raw: str) -> tuple:
    match = _TITLE_YEAR.match(str(raw or "").strip())
    if match:
        title, year = match.group(1).strip(), match.group(2)
    else:
        title, year = str(raw or "").strip(), None

    article = _TRAILING_ARTICLE.match(title)
    if article:
        title = f"{article.group(2)} {article.group(1)}"
    return title, year


def map_to_corpus(source) -> dict:
    """MovieLens movieId -> corpus row.

    The official release ships links.csv, which maps straight onto TMDB
    ids and is used when present. Otherwise films are matched on
    normalised title and year against the corpus - entirely locally, with
    no API calls, and requiring the year to agree so that remakes do not
    silently collapse onto each other.
    """
    import pandas as pd
    from src.tmdb import normalise_title

    kind, path = source

    if kind == "csv" and (path / "links.csv").exists():
        links = pd.read_csv(path / "links.csv", usecols=["movieId", "tmdbId"])
        links = links.dropna(subset=["tmdbId"])
        wanted = corpus.index_by_id()
        mapping = {}
        for movie_id, tmdb_id in zip(links["movieId"].to_numpy(np.int64),
                                     links["tmdbId"].to_numpy(np.int64)):
            row = wanted.get(int(tmdb_id))
            if row is not None:
                mapping[int(movie_id)] = row
        log.info("mapped %d films via links.csv", len(mapping))
        return mapping

    by_title_year, ambiguous = {}, set()
    for row, film in enumerate(corpus.movies()):
        key = (normalise_title(film.get("title")), str(film.get("year") or ""))
        if key in by_title_year:
            ambiguous.add(key)
        by_title_year[key] = row

    mapping = {}
    titles = load_movie_titles(source)
    for movie_id, raw in zip(titles["movieId"].to_numpy(np.int64),
                             titles["title"].astype(str).to_numpy()):
        title, year = _split_title_year(raw)
        key = (normalise_title(title), year or "")
        if key in ambiguous:
            continue                    # two corpus films share it; skip
        row = by_title_year.get(key)
        if row is not None:
            mapping[int(movie_id)] = row

    log.info("mapped %d MovieLens titles onto %d corpus films (by title+year)",
             len(mapping), corpus.size())
    return mapping


def build_matrix(source, mapping: dict, chunk: int = 2_000_000) -> sparse.csr_matrix:
    """Sparse (corpus row x user) matrix of strong positive ratings.

    Only ratings at or above the like threshold are kept. A 2-star rating
    is information, but it is a different kind of information, and mixing
    "watched" with "loved" into one implicit signal is what makes naive
    collaborative filtering recommend whatever is merely popular.
    """
    import pandas as pd

    reader = iter_ratings(source, chunk)

    # Mapped and filtered per chunk with pandas rather than row by row -
    # this is tens of millions of ratings, and a Python loop over them
    # takes minutes where a vectorised map takes seconds.
    parts, total, kept = [], 0, 0
    for part in reader:
        total += len(part)
        part = part[part["rating"] >= config.COLLAB_MIN_RATING]
        part = part.assign(row=part["movieId"].map(mapping))
        part = part.dropna(subset=["row"])
        if len(part):
            parts.append(part[["userId", "row", "rating"]])
            kept += len(part)
        log.info("  scanned %d ratings, kept %d", total, kept)

    if not parts:
        raise RuntimeError("no MovieLens ratings matched the corpus")

    ratings = pd.concat(parts, ignore_index=True)
    user_codes, unique_users = pd.factorize(ratings["userId"].to_numpy())

    # 3.5 -> 1.0, 5.0 -> 2.0: loved counts for more than merely liked.
    span = max(5.0 - config.COLLAB_MIN_RATING, 0.5)
    values = (1.0 + (ratings["rating"].to_numpy(dtype=np.float32)
                     - config.COLLAB_MIN_RATING) / span).astype(np.float32)

    matrix = sparse.csr_matrix(
        (values,
         (ratings["row"].to_numpy(dtype=np.int32),
          user_codes.astype(np.int32))),
        shape=(corpus.size(), len(unique_users)))
    log.info("matrix: %d films x %d users, %d ratings",
             matrix.shape[0], matrix.shape[1], matrix.nnz)
    return matrix


def damp_popularity(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Down-weight prolific users and blockbusters.

    Without this the top neighbour of everything is whatever is most
    rated. A user who rated 3000 films says much less per film than one
    who rated 30, and the same is true of a film everyone has seen.
    """
    matrix = matrix.tocsc()
    per_user = np.asarray(matrix.sum(axis=0)).reshape(-1)
    user_weight = 1.0 / np.sqrt(np.maximum(per_user, 1.0))
    matrix = matrix @ sparse.diags(user_weight.astype(np.float32))

    matrix = matrix.tocsr()
    per_film = np.asarray(matrix.sum(axis=1)).reshape(-1)
    film_weight = 1.0 / np.sqrt(np.maximum(per_film, 1.0))
    return sparse.diags(film_weight.astype(np.float32)) @ matrix


def factorise(matrix: sparse.csr_matrix, factors: int = None) -> tuple:
    """Reduce the user space to dense factors, so scoring is a matmul."""
    factors = factors or config.COLLAB_FACTORS
    factors = min(factors, min(matrix.shape) - 1)

    log.info("running truncated SVD to %d factors", factors)
    started = time.time()
    svd = TruncatedSVD(n_components=factors, algorithm="randomized",
                       n_iter=5, random_state=0)
    vectors = svd.fit_transform(matrix)
    log.info("SVD in %.0fs, explained variance %.2f",
             time.time() - started, float(svd.explained_variance_ratio_.sum()))

    rated = np.asarray((matrix != 0).sum(axis=1)).reshape(-1)
    covered = rated >= config.COLLAB_MIN_RATINGS_PER_FILM

    vectors = normalize(np.asarray(vectors, dtype=np.float32))
    vectors[~covered] = 0.0
    return vectors, covered


def build(upgrade: bool = False) -> CollabSignal:
    source = download(upgrade=upgrade)
    mapping = map_to_corpus(source)
    if not mapping:
        raise RuntimeError("no MovieLens films matched the corpus")
    matrix = build_matrix(source, mapping)
    vectors, covered = factorise(damp_popularity(matrix))
    return CollabSignal(vectors, covered)


def save(signal: CollabSignal, path=None) -> None:
    path = path or config.COLLAB_FILE
    ids = np.array([m["id"] for m in corpus.movies()], dtype=np.int64)
    np.savez_compressed(path, vectors=signal.vectors,
                        covered=signal.covered, ids=ids)
    log.info("saved collaborative factors to %s", path)


def load_collab(path=None) -> CollabSignal | None:
    """Load the factors, realigned onto whatever the corpus is now."""
    path = path or config.COLLAB_FILE
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return None

    by_id = {int(m): i for i, m in enumerate(data["ids"])}
    saved_vectors, saved_covered = data["vectors"], data["covered"]

    rows = corpus.movies()
    vectors = np.zeros((len(rows), saved_vectors.shape[1]), dtype=np.float32)
    covered = np.zeros(len(rows), dtype=bool)
    for i, m in enumerate(rows):
        j = by_id.get(m["id"])
        if j is not None and saved_covered[j]:
            vectors[i] = saved_vectors[j]
            covered[i] = True
    return CollabSignal(vectors, covered)


def main():
    parser = argparse.ArgumentParser(description="Build the collaborative signal")
    parser.add_argument("--factors", type=int, default=config.COLLAB_FACTORS)
    parser.add_argument("--upgrade", action="store_true",
                        help="re-try the official GroupLens dataset and "
                             "rebuild from it if it is reachable")
    parser.add_argument("--check", action="store_true",
                        help="report which dataset is in use and whether the "
                             "official source is reachable, then exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.COLLAB_FACTORS = args.factors

    if args.check:
        provenance = read_provenance()
        found = find_dataset()
        ok, detail = official_available()
        print("\nMovieLens status")
        print(f"  on disk:            {found[1] if found else 'nothing'}")
        print(f"  source:             {provenance.get('source', 'unknown')}")
        print(f"  official reachable: {'yes' if ok else 'no'} ({detail})")
        if ok and provenance.get("source") != "official":
            print("\n  The official dataset is available and you are not on it.")
            print("  Upgrade with:  python -m src.collab --upgrade")
        return

    signal = build(upgrade=args.upgrade)
    save(signal)
    provenance = read_provenance()
    print(f"\nCollaborative signal built")
    print(f"  dataset:       {provenance.get('source', 'unknown')}")
    print(f"  films covered: {signal.covered.sum()} / {corpus.size()} "
          f"({signal.coverage:.1%})")
    print(f"  factors:       {signal.vectors.shape[1]}")
    print(f"  saved to:      {config.COLLAB_FILE}")


if __name__ == "__main__":
    main()
