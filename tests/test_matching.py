"""Tests for the parts that silently returned nonsense.

Every test here corresponds to a real failure: "movies like Interstellar"
came back with Little Women, Unfaithful and Prisoners because the named
film was never looked up, sparse documentary rows outranked the corpus,
and nothing checked whether the winning score meant anything.
"""
import numpy as np
import pytest

from src import config
from src.taste import (
    build_reference_vector,
    build_soup,
    build_taste_vector,
    explain_match,
    fit_vectorizer,
    soup_tokens,
)
from src.tmdb import _score_candidate, normalise_title, overview_terms


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def movie(title, genres, keywords=(), cast=(), director="", overview_terms_=(), **kw):
    return {
        "id": abs(hash(title)) % 10**6,
        "title": title,
        "year": kw.get("year", "2010"),
        "decade": kw.get("decade", "2010s"),
        "genres": list(genres),
        "keywords": list(keywords),
        "cast": list(cast),
        "director": director,
        "directors": [director] if director else [],
        "writers": list(kw.get("writers", [])),
        "collection": kw.get("collection", ""),
        "overview_terms": list(overview_terms_),
        "overview": kw.get("overview", ""),
        "vote_average": kw.get("vote_average", 7.0),
        "vote_count": kw.get("vote_count", 1000),
    }


INTERSTELLAR = movie(
    "Interstellar", ["Science Fiction", "Drama", "Adventure"],
    keywords=["space", "wormhole", "time dilation", "father daughter relationship"],
    cast=["Matthew McConaughey", "Anne Hathaway"], director="Christopher Nolan",
    overview_terms_=["astronaut", "wormhole", "survival", "space", "planet"],
)
ARRIVAL = movie(
    "Arrival", ["Science Fiction", "Drama", "Mystery"],
    keywords=["space", "alien", "linguistics", "time dilation"],
    cast=["Amy Adams"], director="Denis Villeneuve",
    overview_terms_=["linguist", "alien", "space", "communication"],
)
LITTLE_WOMEN = movie(
    "Little Women", ["Drama", "Romance"],
    keywords=["sisters", "19th century", "coming of age"],
    cast=["Saoirse Ronan"], director="Greta Gerwig",
    overview_terms_=["sisters", "marriage", "novel", "family"],
)
UNFAITHFUL = movie(
    "Unfaithful", ["Thriller", "Drama", "Romance"],
    keywords=["infidelity", "affair", "marriage"],
    cast=["Richard Gere"], director="Adrian Lyne",
    overview_terms_=["affair", "husband", "suburban", "betrayal"],
)
SPARSE_DOC = movie("Some Documentary", ["Documentary"])   # nothing else at all

# Filler so the vectoriser has a realistic vocabulary. With only three or
# four films, min_df=2 collapses the vocabulary to a handful of terms and
# everything looks similar to everything else - an artefact of the fixture,
# not of the ranking.
FILLER = [
    movie("Heat", ["Crime", "Thriller", "Action"], keywords=["heist", "los angeles"],
          cast=["Al Pacino"], director="Michael Mann",
          overview_terms_=["detective", "robbery", "crew", "bank"]),
    movie("Toy Story", ["Animation", "Family", "Comedy"], keywords=["toy", "friendship"],
          cast=["Tom Hanks"], director="John Lasseter",
          overview_terms_=["toys", "cowboy", "playroom", "rivalry"]),
    movie("The Shining", ["Horror", "Thriller"], keywords=["hotel", "isolation"],
          cast=["Jack Nicholson"], director="Stanley Kubrick",
          overview_terms_=["hotel", "winter", "caretaker", "madness"]),
    movie("Notting Hill", ["Romance", "Comedy"], keywords=["london", "bookshop"],
          cast=["Hugh Grant"], director="Roger Michell",
          overview_terms_=["bookshop", "actress", "london", "romance"]),
    movie("Gladiator", ["Action", "Drama", "Adventure"], keywords=["rome", "revenge"],
          cast=["Russell Crowe"], director="Ridley Scott",
          overview_terms_=["general", "rome", "arena", "revenge"]),
    movie("Gravity", ["Science Fiction", "Thriller", "Drama"],
          keywords=["space", "survival", "astronaut"], cast=["Sandra Bullock"],
          director="Alfonso Cuaron",
          overview_terms_=["astronaut", "space", "survival", "orbit"]),
    movie("Pride and Prejudice", ["Drama", "Romance"],
          keywords=["19th century", "marriage"], cast=["Keira Knightley"],
          director="Joe Wright",
          overview_terms_=["sisters", "marriage", "estate", "novel"]),
    movie("Se7en", ["Crime", "Thriller", "Mystery"], keywords=["serial killer"],
          cast=["Brad Pitt"], director="David Fincher",
          overview_terms_=["detective", "killer", "sins", "rain"]),
    movie("Up", ["Animation", "Family", "Adventure"], keywords=["balloon", "friendship"],
          cast=["Ed Asner"], director="Pete Docter",
          overview_terms_=["balloons", "house", "widower", "scout"]),
    movie("Casablanca", ["Drama", "Romance", "War"], keywords=["morocco", "wartime"],
          cast=["Humphrey Bogart"], director="Michael Curtiz",
          overview_terms_=["nightclub", "wartime", "refugees", "letters"]),
]

CORPUS = [INTERSTELLAR, ARRIVAL, LITTLE_WOMEN, UNFAITHFUL, SPARSE_DOC] + FILLER


@pytest.fixture
def fitted():
    weights = config.SOUP_WEIGHTS
    vectorizer, matrix = fit_vectorizer([build_soup(m, weights) for m in CORPUS])
    return vectorizer, matrix, weights


# --------------------------------------------------------------------------
# the headline bug: a named film must actually drive the ranking
# --------------------------------------------------------------------------

def test_reference_vector_ranks_arrival_above_little_women(fitted):
    """The original failure, in one assertion."""
    vectorizer, matrix, weights = fitted
    query = build_reference_vector(vectorizer, [INTERSTELLAR], weights)

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(query, matrix)[0]
    by_title = {m["title"]: s for m, s in zip(CORPUS, sims)}

    assert by_title["Arrival"] > by_title["Little Women"]
    assert by_title["Arrival"] > by_title["Unfaithful"]
    assert by_title["Arrival"] > by_title["Some Documentary"]


def test_reference_vector_is_none_without_any_signal(fitted):
    vectorizer, _, weights = fitted
    assert build_reference_vector(vectorizer, [], weights, extra_keywords=[]) is None


def test_keywords_alone_build_a_query_vector(fitted):
    """A vibe request with no named film still has to produce a query."""
    vectorizer, _, weights = fitted
    query = build_reference_vector(vectorizer, [], weights,
                                   extra_keywords=["space", "alien"])
    assert query is not None and query.any()


def test_named_film_outweighs_loose_keywords(fitted):
    """Keywords sharpen the match; they must not override the named film."""
    vectorizer, matrix, weights = fitted
    from sklearn.metrics.pairwise import cosine_similarity

    query = build_reference_vector(vectorizer, [INTERSTELLAR], weights,
                                   extra_keywords=["sisters", "marriage"])
    sims = cosine_similarity(query, matrix)[0]
    by_title = {m["title"]: s for m, s in zip(CORPUS, sims)}
    assert by_title["Arrival"] > by_title["Little Women"]


# --------------------------------------------------------------------------
# sparse rows must not win on shortness alone
# --------------------------------------------------------------------------

def test_sparse_documentary_is_below_the_token_floor():
    assert soup_tokens(SPARSE_DOC) < config.MIN_SOUP_TOKENS
    assert soup_tokens(INTERSTELLAR) >= config.MIN_SOUP_TOKENS


def test_overview_terms_give_thin_rows_real_content():
    bare = movie("Bare", ["Documentary"])
    enriched = movie("Enriched", ["Documentary"],
                     overview_terms_=["climber", "mountain", "ascent", "rope",
                                      "summit", "expedition"])
    assert soup_tokens(enriched) > soup_tokens(bare)


# --------------------------------------------------------------------------
# title resolution: a wrong bind poisons the whole taste vector
# --------------------------------------------------------------------------

def test_exact_title_and_year_beats_a_popular_near_match():
    want = normalise_title("Baby Reindeer")
    exact = {"title": "Baby Reindeer", "release_date": "2024-04-11",
             "vote_count": 50, "popularity": 5.0}
    decoy = {"title": "A Baby Reindeer's First Christmas",
             "release_date": "2020-12-01", "vote_count": 4000, "popularity": 90.0}
    assert _score_candidate(exact, want, "2024") > _score_candidate(decoy, want, "2024")


def test_unrelated_title_is_rejected_outright():
    want = normalise_title("Adolescence")
    unrelated = {"title": "Wicked", "release_date": "2024-11-20",
                 "vote_count": 9000, "popularity": 500.0}
    assert _score_candidate(unrelated, want, "2025") < config.MIN_SEARCH_SCORE


def test_accents_and_punctuation_do_not_block_a_match():
    assert normalise_title("WALL·E") == normalise_title("Wall E")
    assert normalise_title("Amélie") == normalise_title("Amelie")


def test_off_by_one_year_still_matches():
    want = normalise_title("Dune")
    candidate = {"title": "Dune", "release_date": "2021-09-15",
                 "vote_count": 11000, "popularity": 200.0}
    assert _score_candidate(candidate, want, "2020") >= config.MIN_SEARCH_SCORE


# --------------------------------------------------------------------------
# grounded explanations
# --------------------------------------------------------------------------

def test_explain_match_reports_only_real_overlaps():
    why = explain_match(ARRIVAL, reference_features=[INTERSTELLAR])
    assert "Science Fiction" in why["genres"]
    assert "time dilation" in why["keywords"]
    assert "director" not in why          # Villeneuve is not Nolan


def test_explain_match_is_empty_for_an_unrelated_film():
    why = explain_match(UNFAITHFUL, reference_features=[INTERSTELLAR])
    assert not why.get("keywords")
    assert not why.get("director")


def test_explain_match_credits_a_shared_director():
    tenet = movie("Tenet", ["Science Fiction"], director="Christopher Nolan")
    why = explain_match(tenet, reference_features=[INTERSTELLAR])
    assert why["director"] == "Christopher Nolan"


# --------------------------------------------------------------------------
# taste vector arithmetic
# --------------------------------------------------------------------------

def test_rating_weights_give_five_stars_real_pull():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    taste = build_taste_vector(vectors, [5.0, 3.0])[0]
    assert taste[0] > taste[1]


def test_disliked_films_are_subtracted():
    liked = np.array([[1.0, 1.0]])
    disliked = np.array([[0.0, 1.0]])
    taste = build_taste_vector(liked, [5.0], disliked)[0]
    assert taste[0] > taste[1]
    assert (taste >= 0).all()             # clipped, never negative


def test_empty_liked_set_raises_rather_than_returning_garbage():
    with pytest.raises(ValueError):
        build_taste_vector(np.empty((0, 3)), [])


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def test_overview_terms_drop_stopwords_and_short_words():
    terms = overview_terms("The young man must find a way to save his world")
    assert "the" not in terms and "man" not in terms


def test_build_soup_survives_a_legacy_row_with_missing_fields():
    """Rows cached before the schema widened must not crash the vectoriser."""
    legacy = {"id": 1, "title": "Old", "genres": ["Drama"], "keywords": [],
              "cast": [], "director": ""}
    assert isinstance(build_soup(legacy), str)


def test_short_title_does_not_match_a_much_longer_one():
    """The 'Adolescence' failure: containment alone is not a match."""
    want = normalise_title("Adolescence")
    decoy = {"title": "The Real Adolescence: Our Killer Kids",
             "release_date": "2025-01-01", "vote_count": 20, "popularity": 3.0}
    assert _score_candidate(decoy, want, "2025") < config.MIN_SEARCH_SCORE


def test_close_variant_titles_still_match():
    """Tightening containment must not start rejecting real matches."""
    want = normalise_title("Alien")
    aliens = {"title": "Aliens", "release_date": "1986-07-18",
              "vote_count": 7000, "popularity": 60.0}
    assert _score_candidate(aliens, want, "1986") >= config.MIN_SEARCH_SCORE


# --------------------------------------------------------------------------
# score floors
# --------------------------------------------------------------------------

def _signal(films):
    """A metadata signal over exactly the films given.

    Tests need their own small corpus; a signal owns the film list its
    rows align to, so building one here keeps them off the real cache.
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from src.signals import Signal

    weights = config.SOUP_WEIGHTS
    vectorizer, matrix = fit_vectorizer([build_soup(m, weights) for m in films])
    dense = normalize(np.asarray(matrix.todense(), dtype=np.float32))

    class TestSignal(Signal):
        name = "metadata"

        def encode_movie(self, features):
            # The seed film is deliberately outside the fixture corpus,
            # so it has to be vectorised on the fly - the same thing the
            # real metadata signal does for a film nobody has indexed.
            existing = super().encode_movie(features)
            if existing is not None:
                return existing
            soup = build_soup(features, weights)
            return np.asarray(vectorizer.transform([soup]).todense(),
                              dtype=np.float32).reshape(-1)

    return TestSignal(dense, movies=list(films))


def test_weak_picks_are_dropped_rather_than_padding_the_list():
    """A six-item list must not be filled out with things nobody asked for."""
    from src import taste

    corpus = [ARRIVAL, LITTLE_WOMEN, UNFAITHFUL] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    titles = [p["title"] for p in picks]
    assert "Arrival" in titles
    assert "Unfaithful" not in titles
    assert "Notting Hill" not in titles
    assert len(picks) < len(corpus)        # not just the whole corpus back


def test_nothing_is_returned_when_nothing_stands_out():
    """The bar is in standard deviations, so it survives a signal swap.

    A request that matches everything equally badly has no best answer,
    and saying so is better than shipping whatever sorted first.
    """
    from src import taste

    corpus = [LITTLE_WOMEN, UNFAITHFUL] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=3.0)
    assert picks == []


def test_the_top_pick_anchors_where_the_list_stops():
    """A pick far below the best one is a different population."""
    from src import taste

    corpus = [ARRIVAL, LITTLE_WOMEN, UNFAITHFUL] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=-10.0)
    assert picks, "expected at least one pick"
    assert picks[-1]["score"] >= picks[0]["score"] * config.MIN_Z_RETAIN - 1e-6


def test_every_reference_pick_carries_a_real_overlap():
    from src import taste

    corpus = [ARRIVAL, LITTLE_WOMEN, UNFAITHFUL] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert picks, "expected at least one pick"
    for p in picks:
        assert p["why"], f"{p['title']} was recommended with no computable overlap"


def test_a_film_the_user_already_watched_is_never_recommended():
    from src import taste

    corpus = [ARRIVAL, LITTLE_WOMEN, UNFAITHFUL] + FILLER
    profile = {
        "liked": [{"title": "Interstellar", "year": "2010", "rating": 5.0}],
        "disliked": [],
        "watched_keys": {("arrival", "2010")},
    }
    picks = taste.recommend(profile=None, reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert "Arrival" in [p["title"] for p in picks]

    picks = taste.recommend(profile={**profile, "liked": []},
                            reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert "Arrival" not in [p["title"] for p in picks]


def test_the_named_film_is_never_recommended_back():
    from src import taste

    corpus = [INTERSTELLAR, ARRIVAL, LITTLE_WOMEN] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert "Interstellar" not in [p["title"] for p in picks]


def test_a_making_of_documentary_is_not_a_similar_film():
    """"Interstellar: Nolan's Odyssey" was ranking third for "like Interstellar"."""
    from src import taste

    making_of = movie("Interstellar: Nolan's Odyssey", ["Documentary", "TV Movie"],
                      keywords=["space", "wormhole"], director="Christopher Nolan",
                      overview_terms_=["astronaut", "wormhole", "space", "filming"])
    corpus = [ARRIVAL, LITTLE_WOMEN, making_of] + FILLER
    picks = taste.recommend(reference_features=[INTERSTELLAR],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert "Interstellar: Nolan's Odyssey" not in [p["title"] for p in picks]


def test_documentary_reference_still_returns_documentaries():
    """The mode guard must work in both directions, not just exclude docs."""
    from src import taste

    seed = movie("Free Solo", ["Documentary"], keywords=["climbing", "el capitan"],
                 overview_terms_=["climber", "rope", "yosemite", "ascent", "summit"])
    other_doc = movie("The Dawn Wall", ["Documentary"], keywords=["climbing", "el capitan"],
                      overview_terms_=["climber", "rope", "yosemite", "ascent", "summit"])
    corpus = [other_doc, LITTLE_WOMEN, ARRIVAL] + FILLER
    picks = taste.recommend(reference_features=[seed],
                            signal=_signal(corpus), top_n=6, min_z=0.5)
    assert "The Dawn Wall" in [p["title"] for p in picks]


# --------------------------------------------------------------------------
# vibe requests must justify themselves too
# --------------------------------------------------------------------------

def test_a_vibe_pick_is_explained_against_the_request():
    """With no named film and no profile, the request itself is the source.

    Otherwise every mood-based recommendation is written up as having no
    computed overlap, which is honest and useless in equal measure.
    """
    from src.taste import explain_match

    why = explain_match(ARRIVAL, query_keywords=["alien", "space"],
                        query_genres=["Science Fiction"])
    assert "Science Fiction" in why["genres"]
    assert any("alien" in k.lower() or "space" in k.lower()
               for k in why["keywords"])


def test_a_vibe_pick_with_nothing_in_common_stays_unexplained():
    from src.taste import explain_match

    why = explain_match(LITTLE_WOMEN, query_keywords=["heist", "spacecraft"],
                        query_genres=["Action"])
    assert why == {}
