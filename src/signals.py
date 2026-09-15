"""Scoring signals: different ways of saying two films are alike.

Every signal exposes the same three things, so the reranker can blend
them and the benchmark can A/B them one at a time:

    vectors   (n_corpus, d)  L2-normalised, row-aligned to src.corpus
    covered   (n_corpus,)    bool - False where the signal has no data
    encode_*  turn a film or a free-text query into a query vector

The three implemented signals answer genuinely different questions:

  metadata   - do these films share explicit facts? (director, franchise,
               billed cast). Precise, and blind to anything unstated.
  semantic   - do these films read as being about the same thing? Handles
               "dark and mind-bending", which no token-overlap model can.
  collab     - do the same people like both? Captures tone and feel that
               appear in no metadata field at all.
"""
import logging

import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import normalize

from src import config, corpus

log = logging.getLogger(__name__)


def _l2(matrix):
    return normalize(matrix, norm="l2", copy=True)


def _dense_vector(vector) -> np.ndarray:
    """One row as a flat dense array, whether it started sparse or not.

    Signal vectors are deliberately not all dense. The metadata matrix is
    5423 x 21843 but only 179k of those cells are non-zero: 1.4MB sparse
    against 474MB dense. Densifying it on load made every process that
    touched the model allocate half a gigabyte for nothing.
    """
    if sp.issparse(vector):
        return np.asarray(vector.todense(), dtype=np.float32).reshape(-1)
    return np.asarray(vector, dtype=np.float32).reshape(-1)


class Signal:
    """Base: vectors row-aligned to the corpus, plus a coverage mask."""

    name = "signal"

    def __init__(self, vectors, covered=None, movies=None):
        self.vectors = vectors
        n = vectors.shape[0]
        self.covered = (np.ones(n, dtype=bool) if covered is None
                        else np.asarray(covered, dtype=bool))
        # A signal owns the film list its rows line up with. Reaching for
        # a module-level corpus instead would make row 400 mean different
        # films in different signals the moment one of them was built
        # against a different snapshot.
        self._movies = movies
        self._index = None

    @property
    def movies(self) -> list:
        if self._movies is None:
            self._movies = corpus.movies()
        return self._movies

    @property
    def index_by_id(self) -> dict:
        if self._index is None:
            self._index = {m["id"]: i for i, m in enumerate(self.movies)}
        return self._index

    def row_of(self, movie_id):
        return self.index_by_id.get(movie_id)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    @property
    def coverage(self) -> float:
        return float(self.covered.mean())

    # -- query construction ------------------------------------------------

    def encode_movie(self, features: dict):
        """Vector for a film, which may or may not be in the corpus."""
        row = self.row_of(features.get("id"))
        if row is not None and self.covered[row]:
            return _dense_vector(self.vectors[row])
        return None

    def encode_query(self, text: str, keywords=None):
        """Vector for a free-text request. Not every signal can do this."""
        return None

    # -- scoring -----------------------------------------------------------

    def similarity(self, query, context: str = "query") -> np.ndarray:
        """Cosine of a query vector against every corpus row.

        Vectors are pre-normalised, so this is a dot product. Rows the
        signal knows nothing about score zero rather than a misleading
        small number. `context` is accepted for interface parity with the
        blend, which weights differently per context; a lone signal has
        nothing to weight, so it ignores it.
        """
        if query is None:
            return None
        q = _dense_vector(query)
        norm = np.linalg.norm(q)
        if norm == 0:
            return None
        sims = np.asarray(self.vectors @ (q / norm)).reshape(-1)
        sims[~self.covered] = 0.0
        return sims

    def similarity_to_row(self, row: int) -> np.ndarray:
        """Similarity of every film to one corpus film."""
        if not self.covered[row]:
            return None
        return self.similarity(_dense_vector(self.vectors[row]))

    def profile_similarity_rows(self, rows, ratings=None, k: int = None):
        """Profile similarity where the liked films are corpus rows."""
        vectors = [_dense_vector(self.vectors[r]) for r in rows if self.covered[r]]
        kept = [rt for r, rt in zip(rows, ratings or [])
                if self.covered[r]] if ratings else None
        return self.profile_similarity(vectors, kept, k)

    def profile_similarity(self, liked_vectors, ratings=None,
                           k: int = None) -> np.ndarray:
        """Score every film against a set of liked films.

        This is idea #2. The old model averaged every liked film into one
        centroid, which is the wrong shape for real taste: someone who
        watches horror and rom-coms gets a centroid that is neither, and
        the recommendations land in the empty space between their actual
        interests.

        Instead each candidate is scored by how well it matches the few
        liked films it matches best. A horror fan's horror picks are
        judged against their horror films, and their rom-com picks
        against their rom-coms, so both clusters survive.
        """
        vectors = [v for v in liked_vectors if v is not None]
        if not vectors:
            return None

        k = k or config.TASTE_TOP_K
        stacked = _l2(np.vstack([_dense_vector(v).reshape(1, -1) for v in vectors]))

        weights = None
        if ratings is not None and len(ratings) == stacked.shape[0]:
            weights = np.array([max(float(r) - 2.5, 0.1) for r in ratings])

        # (n_corpus, n_liked)
        sims = np.asarray(self.vectors @ stacked.T)
        if sims.ndim == 1:
            sims = sims.reshape(-1, 1)

        # Top-K: how well does this match the few films it matches best?
        k = min(k, sims.shape[1])
        top_k = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)

        # Centroid: how well does this match the profile as a whole?
        centre = np.average(stacked, axis=0, weights=weights)
        norm = np.linalg.norm(centre)
        centroid = (self.vectors @ (centre / norm)) if norm else np.zeros_like(top_k)
        centroid = np.asarray(centroid).reshape(-1)

        alpha = config.TASTE_CENTROID_WEIGHT
        scores = alpha * centroid + (1.0 - alpha) * top_k
        scores[~self.covered] = 0.0
        return scores


# --------------------------------------------------------------------------
# metadata: TF-IDF over the token soup
# --------------------------------------------------------------------------

class MetadataSignal(Signal):
    """Explicit shared facts: director, franchise, billed cast, genres.

    This is the original model, kept because it is genuinely good at what
    it does. Nothing beats it for "same director" or "same franchise",
    because those are exact token matches rather than a matter of degree.
    Its blind spot is everything nobody wrote down.
    """

    name = "metadata"

    def __init__(self, taste_model):
        from src.taste import build_soup

        self._model = taste_model
        self._build_soup = build_soup
        self.weights = taste_model.weights or config.SOUP_WEIGHTS

        rows = corpus.movies()
        model_rows = {m["id"]: i for i, m in enumerate(taste_model.movies)}

        order, covered = [], np.zeros(len(rows), dtype=bool)
        for i, m in enumerate(rows):
            j = model_rows.get(m["id"])
            if j is None:
                order.append(0)          # placeholder; masked out below
            else:
                order.append(j)
                covered[i] = True

        vectors = _l2(taste_model.matrix[np.array(order, dtype=int)]).tocsr()
        # Zero the placeholder rows without densifying the matrix.
        if (~covered).any():
            keep = sp.diags(covered.astype(np.float32))
            vectors = (keep @ vectors).tocsr()
        super().__init__(vectors, covered)

    def encode_movie(self, features: dict):
        existing = super().encode_movie(features)
        if existing is not None:
            return existing
        # A film outside the corpus can still be vectorised on the fly.
        soup = self._build_soup(features, self.weights)
        return _dense_vector(self._model.vectorizer.transform([soup]))

    def encode_query(self, text: str, keywords=None):
        """Only the extracted keywords are usable here, not the raw text.

        A TF-IDF vocabulary built from genre and keyword tokens has no
        entry for "mind-bending", so the free-text request has to be
        translated into vocabulary terms before it means anything. That
        lossy translation is exactly what the semantic signal removes.
        """
        terms = [str(k).lower().replace(" ", "") for k in (keywords or []) if k]
        if not terms:
            return None
        vector = _dense_vector(self._model.vectorizer.transform([" ".join(terms)]))
        return vector if vector.any() else None


# --------------------------------------------------------------------------
# semantic: dense embeddings of a written description
# --------------------------------------------------------------------------

def movie_document(features: dict) -> str:
    """The prose a film gets embedded as.

    Ordered most-distinctive first, because static embeddings average
    their tokens and a long tail of cast names would drown the plot.
    """
    parts = [
        features.get("title", ""),
        features.get("tagline", ""),
        features.get("overview", ""),
        ", ".join(features.get("genres", [])),
        ", ".join(features.get("keywords", [])[:12]),
    ]
    director = features.get("director")
    if director:
        parts.append(f"Directed by {director}.")
    cast = features.get("cast", [])[:4]
    if cast:
        parts.append("Starring " + ", ".join(cast) + ".")
    return " ".join(p for p in parts if p)


class SemanticSignal(Signal):
    """What the film is *about*, in a space where paraphrases are close.

    This is what makes a free-text request work at all. "Dark and
    mind-bending" is not a token in any metadata vocabulary, so the old
    pipeline had to ask an LLM to rewrite it as TMDB keywords and then
    hope those keywords were literally present on the right films. Here
    the request is embedded directly and compared to what each film is
    described as being about.
    """

    name = "semantic"

    def __init__(self, vectors, covered=None, encoder=None, movies=None):
        super().__init__(np.asarray(vectors, dtype=np.float32), covered, movies)
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = load_encoder()
        return self._encoder

    def encode_movie(self, features: dict):
        existing = super().encode_movie(features)
        if existing is not None:
            return existing
        return self.encoder.encode([movie_document(features)])[0]

    def encode_query(self, text: str, keywords=None):
        text = (text or "").strip()
        if keywords:
            text = f"{text}. {', '.join(str(k) for k in keywords)}"
        if not text:
            return None
        return self.encoder.encode([text])[0]


_encoder = None


def load_encoder():
    global _encoder
    if _encoder is None:
        from model2vec import StaticModel
        log.info("loading embedding model %s", config.EMBED_MODEL)
        _encoder = StaticModel.from_pretrained(config.EMBED_MODEL)
    return _encoder


def build_semantic(batch: int = 512) -> SemanticSignal:
    """Embed the whole corpus. Cheap enough to redo whenever it changes."""
    encoder = load_encoder()
    rows = corpus.movies()
    docs = [movie_document(m) for m in rows]

    chunks = []
    for start in range(0, len(docs), batch):
        chunks.append(np.asarray(encoder.encode(docs[start:start + batch]),
                                 dtype=np.float32))
    vectors = _l2(np.vstack(chunks))

    # A film with no overview and no keywords embeds to almost nothing
    # useful; better to abstain than to contribute noise to the blend.
    covered = np.array([bool((m.get("overview") or "").strip()
                             or m.get("keywords")) for m in rows])
    vectors[~covered] = 0.0
    return SemanticSignal(vectors, covered, encoder)


def save_semantic(signal: SemanticSignal, path=None) -> None:
    path = path or config.EMBED_FILE
    ids = np.array([m["id"] for m in corpus.movies()], dtype=np.int64)
    np.savez_compressed(path, vectors=signal.vectors,
                        covered=signal.covered, ids=ids)
    log.info("saved %d embeddings to %s", signal.vectors.shape[0], path)


def load_semantic(path=None) -> SemanticSignal | None:
    """Load cached embeddings, realigning them onto the current corpus."""
    path = path or config.EMBED_FILE
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception as exc:
        log.warning("could not read embeddings at %s: %s", path, exc)
        return None

    saved_ids = data["ids"]
    saved_vectors, saved_covered = data["vectors"], data["covered"]
    by_id = {int(mid): i for i, mid in enumerate(saved_ids)}

    rows = corpus.movies()
    dim = saved_vectors.shape[1]
    vectors = np.zeros((len(rows), dim), dtype=np.float32)
    covered = np.zeros(len(rows), dtype=bool)

    for i, m in enumerate(rows):
        j = by_id.get(m["id"])
        if j is not None and saved_covered[j]:
            vectors[i] = saved_vectors[j]
            covered[i] = True

    missing = (~covered).sum()
    if missing:
        log.info("%d corpus films have no cached embedding", missing)
    return SemanticSignal(vectors, covered)
