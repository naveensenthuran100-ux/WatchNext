"""Build and cache dense embeddings for the corpus.

    python -m src.embed          # embed every corpus film, save to data/

Static distilled embeddings (model2vec) rather than a transformer: no
torch, ~30MB on disk, and encoding the whole corpus takes seconds on a
CPU. For a few thousand films the nearest-neighbour search is a single
numpy matmul, so there is no vector database to run either.
"""
import argparse
import logging
import time

from src import config, corpus, signals

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Embed the film corpus")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    started = time.time()
    signal = signals.build_semantic(batch=args.batch)
    signals.save_semantic(signal)

    print(f"\nEmbedded {len(corpus.movies())} films in {time.time() - started:.1f}s")
    print(f"  model:    {config.EMBED_MODEL}")
    print(f"  dim:      {signal.vectors.shape[1]}")
    print(f"  coverage: {signal.coverage:.1%}")
    print(f"  saved to: {config.EMBED_FILE}")


if __name__ == "__main__":
    main()
