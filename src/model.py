"""The trained artifact: a fitted TF-IDF vectorizer plus the corpus matrix.

Fitting the vectorizer per request (what the old code did) meant liked
films and candidates were vectorised in whatever vocabulary that one
request happened to produce, and the work was redone every time. Fitting
once over a fixed corpus and saving it makes scoring a single sparse
matrix multiply, and makes results comparable between runs.
"""
import logging
from dataclasses import dataclass, field

import joblib

from src import config

log = logging.getLogger(__name__)


@dataclass
class TasteModel:
    vectorizer: object
    matrix: object                  # sparse (n_movies x n_terms), L2-normalised
    movies: list                    # feature dicts, row-aligned with matrix
    weights: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def index_by_id(self) -> dict:
        if not hasattr(self, "_index"):
            self._index = {m["id"]: i for i, m in enumerate(self.movies)}
        return self._index

    def __len__(self) -> int:
        return len(self.movies)


def save(model: TasteModel, path=None) -> None:
    path = path or config.MODEL_FILE
    joblib.dump(model, path, compress=3)
    log.info("saved model with %d movies to %s", len(model), path)


def load(path=None) -> TasteModel | None:
    path = path or config.MODEL_FILE
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        log.warning("could not load model at %s: %s", path, exc)
        return None
