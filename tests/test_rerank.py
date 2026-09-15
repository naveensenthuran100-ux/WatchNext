"""Tests for the LLM reranking stage.

The reranker is allowed to reorder and trim an already-vetted shortlist,
and nothing more. These pin down the boundary that keeps it from
reintroducing the original confabulation bug: it can never surface a film
that was not a candidate, and any failure must fall back to the ranking
system's own order rather than to nothing.
"""
import pytest

from src import recommender


def _candidates(n):
    return [{"id": i, "title": f"Film {i}", "year": "2010",
             "overview": f"plot {i}", "why": {"genres": ["Drama"]}}
            for i in range(n)]


def test_reorders_within_the_shortlist(monkeypatch):
    monkeypatch.setattr(recommender, "ask_ai",
                        lambda *a, **k: '{"order": [4, 2, 0]}')
    picks = recommender.rerank_picks(_candidates(10), "something good", n=3)
    assert [p["id"] for p in picks] == [4, 2, 0]


def test_never_surfaces_a_film_that_was_not_a_candidate(monkeypatch):
    # 99 is not a valid index; it must be dropped, not fabricated.
    monkeypatch.setattr(recommender, "ask_ai",
                        lambda *a, **k: '{"order": [99, 1, 2]}')
    cands = _candidates(10)
    picks = recommender.rerank_picks(cands, "x", n=3)
    assert all(p in cands for p in picks)
    assert [p["id"] for p in picks] == [1, 2, 0]  # 99 dropped, topped up with best remaining


def test_falls_back_to_blend_order_on_bad_json(monkeypatch):
    monkeypatch.setattr(recommender, "ask_ai", lambda *a, **k: "not json at all")
    picks = recommender.rerank_picks(_candidates(10), "x", n=3)
    assert [p["id"] for p in picks] == [0, 1, 2]


def test_falls_back_to_blend_order_when_model_is_down(monkeypatch):
    monkeypatch.setattr(recommender, "ask_ai", lambda *a, **k: None)
    picks = recommender.rerank_picks(_candidates(10), "x", n=3)
    assert [p["id"] for p in picks] == [0, 1, 2]


def test_always_returns_a_full_list_from_a_partial_answer(monkeypatch):
    # Model returns only one valid pick; the list must still fill to n
    # from the vetted shortlist, never with anything from outside it.
    monkeypatch.setattr(recommender, "ask_ai", lambda *a, **k: '{"order": [7]}')
    cands = _candidates(10)
    picks = recommender.rerank_picks(cands, "x", n=3)
    assert len(picks) == 3
    assert picks[0]["id"] == 7
    assert all(p in cands for p in picks)


def test_duplicates_in_the_answer_are_ignored(monkeypatch):
    monkeypatch.setattr(recommender, "ask_ai",
                        lambda *a, **k: '{"order": [3, 3, 3, 1]}')
    picks = recommender.rerank_picks(_candidates(10), "x", n=3)
    ids = [p["id"] for p in picks]
    assert len(ids) == len(set(ids))
    assert ids[:2] == [3, 1]


def test_a_shortlist_at_or_below_n_is_returned_untouched(monkeypatch):
    # No model call should even happen: there is nothing to choose.
    called = False
    def boom(*a, **k):
        nonlocal called
        called = True
        return '{"order": [1, 0]}'
    monkeypatch.setattr(recommender, "ask_ai", boom)
    cands = _candidates(3)
    picks = recommender.rerank_picks(cands, "x", n=3)
    assert picks == cands
    assert not called, "must not spend an LLM call when there is nothing to rerank"
