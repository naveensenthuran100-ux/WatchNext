"""Blending several signals into one score.

No single signal is right on its own. Metadata is precise about facts and
blind to everything else; embeddings understand what a film is about but
happily match a documentary to the feature it documents; collaborative
factors capture tone that appears in no field anywhere, but only for
films enough people have rated.

Blending them means a film has to look right by more than one measure,
which is what kills the failure this project started with - a pick that
scored well by one weak signal and was then written up as if the
connection were real.

Coverage is renormalised per film. A film MovieLens has never heard of is
scored on the signals that do know it, rather than being penalised for
data we never had.
"""
import logging

import numpy as np

from src import config, corpus

log = logging.getLogger(__name__)


class BlendedSignal:
    """Several signals under the Signal interface, so anything can use it."""

    name = "blend"

    def __init__(self, signals, weights: dict = None, query_weights: dict = None):
        if not signals:
            raise ValueError("a blend needs at least one signal")
        self.signals = list(signals)
        # Two profiles, because a named film and a typed mood are answered
        # from different signals. `weights` leads whenever a film anchors
        # the request (item-to-item and taste-profile scoring); the far
        # more semantic `query_weights` takes over for free text. See
        # config.SIGNAL_WEIGHTS for the measured split.
        self.weights = dict(weights or config.SIGNAL_WEIGHTS)
        self.query_weights = dict(
            query_weights if query_weights is not None
            else config.QUERY_SIGNAL_WEIGHTS)

        # Covered where at least one member knows the film.
        stacked = np.vstack([s.covered for s in self.signals])
        self.covered = stacked.any(axis=0)

        sizes = {len(s.covered) for s in self.signals}
        if len(sizes) > 1:
            raise ValueError(f"signals disagree on corpus size: {sizes}")

    @property
    def movies(self) -> list:
        return self.signals[0].movies

    def row_of(self, movie_id):
        return self.signals[0].row_of(movie_id)

    def __len__(self) -> int:
        return len(self.covered)

    @property
    def coverage(self) -> float:
        return float(self.covered.mean())

    def weight_of(self, signal, profile: dict = None) -> float:
        return float((profile or self.weights).get(signal.name, 0.0))

    # -- query construction ------------------------------------------------

    def encode_movie(self, features: dict) -> dict:
        return {s.name: s.encode_movie(features) for s in self.signals}

    def encode_query(self, text: str, keywords=None) -> dict:
        return {s.name: s.encode_query(text, keywords) for s in self.signals}

    # -- scoring -----------------------------------------------------------

    def _standardise(self, sims, covered) -> np.ndarray | None:
        """Put one signal's scores on a scale comparable to the others.

        Raw cosines are not comparable between signals. Measured on this
        corpus, a metadata score runs 0.005 (median) to 0.16 (best match)
        while a semantic score runs 0.22 to 0.66. Averaging those raw
        numbers does not blend the signals - semantic's baseline alone
        exceeds metadata's entire range, so metadata contributes nothing
        but noise and the blend scores worse than either signal alone.

        Standardising each signal per query fixes that: every signal ends
        up saying "how unusual is this match, for me", which is the thing
        that is actually comparable.
        """
        if sims is None:
            return None
        values = sims[covered]
        if values.size < 2:
            return None
        spread = values.std()
        if spread == 0:
            return None

        out = np.zeros_like(sims, dtype=np.float64)
        out[covered] = (values - values.mean()) / spread
        return out

    def _combine(self, per_signal: list) -> np.ndarray:
        """Weighted mean of standardised scores.

        Weights are renormalised per film over the signals that actually
        cover it, so a film MovieLens has never seen is judged on content
        alone rather than penalised for data we never had.
        """
        contributions = []
        for weight, sims, covered in per_signal:
            if weight <= 0 or sims is None:
                continue
            standardised = self._standardise(sims, covered)
            if standardised is not None:
                contributions.append((weight, standardised, covered))

        if not contributions:
            return None

        total = np.zeros(len(self.covered), dtype=np.float64)
        weight_sum = np.zeros(len(self.covered), dtype=np.float64)
        for weight, sims, covered in contributions:
            total[covered] += weight * sims[covered]
            weight_sum[covered] += weight

        scores = np.zeros_like(total)
        live = weight_sum > 0
        scores[live] = total[live] / weight_sum[live]
        return scores

    def similarity(self, query, context: str = "query") -> np.ndarray:
        """Cosine of a query against the corpus, under a weight profile.

        context="query" (the default) is a typed mood with no film named,
        so the semantic-heavy query profile leads. context="reference" is
        a named film's own vector, which wants the metadata-heavy profile.
        """
        query = query or {}
        profile = self.weights if context == "reference" else self.query_weights
        return self._combine([
            (self.weight_of(s, profile), s.similarity(query.get(s.name)), s.covered)
            for s in self.signals
        ])

    def similarity_to_row(self, row: int) -> np.ndarray:
        """Item-to-item: a named film anchors it, so the reference profile."""
        return self._combine([
            (self.weight_of(s), s.similarity_to_row(row), s.covered)
            for s in self.signals
        ])

    def profile_similarity(self, liked_vectors, ratings=None, k: int = None):
        """liked_vectors is a list of per-signal dicts from encode_movie."""
        per_signal = []
        for s in self.signals:
            vectors = [v.get(s.name) if isinstance(v, dict) else v
                       for v in liked_vectors]
            vectors = [v for v in vectors if v is not None]
            if not vectors:
                continue
            per_signal.append((self.weight_of(s),
                               s.profile_similarity(vectors, ratings, k),
                               s.covered))
        return self._combine(per_signal)

    def profile_similarity_rows(self, rows, ratings=None, k: int = None):
        return self._combine([
            (self.weight_of(s), s.profile_similarity_rows(rows, ratings, k),
             s.covered)
            for s in self.signals
        ])


_default = None


def load_default(include_collab: bool = True, refresh: bool = False):
    """The blend the bot actually uses, skipping anything not built yet.

    Cached: this reads a fitted vectoriser and two matrices off disk, and
    every recommendation would otherwise pay for it again.
    """
    global _default
    if _default is not None and not refresh:
        return _default
    _default = _build_default(include_collab)
    return _default


def _build_default(include_collab: bool = True) -> BlendedSignal | None:
    from src import model as model_mod, signals as sig

    built = []
    taste_model = model_mod.load()
    if taste_model:
        built.append(sig.MetadataSignal(taste_model))

    semantic = sig.load_semantic()
    if semantic is not None:
        built.append(semantic)

    # Signals weighted at zero are skipped rather than loaded and
    # multiplied out, so switching one off costs nothing at runtime.
    if config.SIGNAL_WEIGHTS.get("graph", 0) > 0:
        try:
            from src.graph import load_graph
            graph = load_graph()
            if graph is not None:
                built.append(graph)
        except ImportError:
            pass

    if include_collab:
        try:
            from src.collab import load_collab
            collab = load_collab()
            if collab is not None:
                built.append(collab)
        except ImportError:
            pass

    if not built:
        return None
    log.info("scoring with: %s", ", ".join(s.name for s in built))
    return BlendedSignal(built)
