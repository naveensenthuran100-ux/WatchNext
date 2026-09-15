"""Turns a natural-language message into a recommendation.

The flow is: read the message, resolve any film the user named, rank the
corpus against whatever signals we actually have, then write the result
up using only facts the ranking stage produced.

That last part matters. The write-up stage used to be handed the raw
request plus a list of films chosen by a mechanism that never saw the
request, and asked to explain why they fit - so it confabulated. It is
now given the specific overlaps (shared director, genres, keywords) and
told to use only those.
"""
import json
import logging
from typing import NamedTuple

from groq import Groq

from src import config, feedback
from src.cache import cached_features_many, get_cached_profile, put_profile
from src.letterboxd import get_taste_profile
from src.taste import recommend, resolve_films, resolve_titles
from src.tmdb import similar_movie_ids, tmdb_get

log = logging.getLogger(__name__)

_client = None


def client() -> Groq:
    global _client
    if _client is None:
        config.require("GROQ_API_KEY")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def ask_ai(prompt: str, system: str, json_mode: bool = False) -> str | None:
    """Call Groq, falling back to the secondary model on *any* failure.

    The original only fell through on rate limits, so a bad model id or a
    transient 500 raised straight out to the caller and the user saw
    "something went wrong" with no retry.
    """
    last_error = None
    for model_name in (config.PRIMARY_MODEL, config.FALLBACK_MODEL):
        try:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = client().chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as exc:
            log.warning("model %s failed (%s) - trying next", model_name, exc)
            last_error = exc

    log.error("all models failed: %s", last_error)
    return None


INTENT_SYSTEM = """
You read a chat message and decide whether the person is asking for a film
recommendation, then extract search parameters.

Reply with JSON only, in this exact shape:
{
  "intent": "recommend" | "chat" | "unclear",
  "reference_titles": ["exact film title", ...],
  "genres": ["genre", ...],
  "mood": "one word",
  "keywords": ["keyword", ...],
  "decade": "any" | "80s" | "90s" | "00s" | "10s" | "20s",
  "explanation": "one sentence on how you read the request",
  "reply": "a short friendly reply, used only when intent is chat or unclear"
}

reference_titles is the most important field. Put in it any film the user
names as a comparison point - "like Interstellar", "in the vein of Heat",
"more stuff like Arrival". Give the film's real title only, with no year
and no extra words. Leave it as [] when they name no film.

Do not put a named film's title into keywords. Keywords are for themes
and subjects ("space travel", "time dilation", "heist"), not titles.

Use "chat" for greetings, thanks, small talk and questions about you.
Use "unclear" when they may want a film but you cannot tell what kind.
Use "recommend" only when there is a real mood, genre or reference to work
with. Genres must be TMDB genres (Action, Adventure, Animation, Comedy,
Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music,
Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western).
"""


def interpret_vibe(user_message: str) -> dict | None:
    """Classify intent and extract parameters in a single call."""
    raw = ask_ai(f"Message: {user_message!r}", INTENT_SYSTEM, json_mode=True)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("model returned non-JSON: %r", raw[:200])
        return None

    def clean_list(field: str) -> list:
        return [x for x in parsed.get(field) or [] if x and str(x).upper() != "N/A"]

    genres     = clean_list("genres")
    keywords   = clean_list("keywords")
    references = clean_list("reference_titles")

    # Trust the model's own signal: if it found nothing to search on, this
    # was not a recommendation request no matter what it labelled it.
    intent = parsed.get("intent", "unclear")
    if intent == "recommend" and not genres and not keywords and not references:
        intent = "unclear"

    return {
        "intent":           intent,
        "reference_titles": references,
        "genres":           genres,
        "keywords":         keywords,
        "mood":             parsed.get("mood", ""),
        "decade":           parsed.get("decade", "any"),
        "explanation":      parsed.get("explanation", ""),
        "reply":            parsed.get("reply", ""),
    }


def get_genre_ids(genre_names: list) -> list:
    data = tmdb_get("/genre/movie/list")
    genre_map = {g["name"].lower(): g["id"] for g in data.get("genres", [])}
    return [str(genre_map[n.strip().lower()])
            for n in genre_names if n.strip().lower() in genre_map]


DECADE_RANGES = {
    "80s": ("1980-01-01", "1989-12-31"),
    "90s": ("1990-01-01", "1999-12-31"),
    "00s": ("2000-01-01", "2009-12-31"),
    "10s": ("2010-01-01", "2019-12-31"),
    "20s": ("2020-01-01", "2029-12-31"),
}


def fetch_movies_by_vibe(vibe: dict) -> list:
    """Last-resort genre browse, used when nothing scores above the floor."""
    genre_ids = get_genre_ids(vibe.get("genres", []))

    params = {
        "sort_by":        "vote_average.desc",
        "vote_count.gte": 300,
    }
    if genre_ids:
        params["with_genres"] = ",".join(genre_ids)

    decade = vibe.get("decade", "any")
    if decade in DECADE_RANGES:
        start, end = DECADE_RANGES[decade]
        params["primary_release_date.gte"] = start
        params["primary_release_date.lte"] = end

    return tmdb_get("/discover/movie", params=params).get("results", [])[:3]


def fetch_similar_to(reference_features: list, seen_keys: set = None,
                     limit: int = 3) -> list:
    """TMDB's own neighbours for a named film.

    Used when the named film is real but our corpus has nothing close
    enough to it. Better to widen than to lower the bar.
    """
    ids = []
    for f in reference_features[:2]:
        ids.extend(similar_movie_ids(f["id"], limit=30))

    reference_ids = {f["id"] for f in reference_features}
    movies = cached_features_many([i for i in dict.fromkeys(ids)
                                   if i not in reference_ids][:40])

    seen_keys = seen_keys or set()
    movies = [m for m in movies
              if (str(m.get("title", "")).lower().strip(),
                  str(m.get("year")) if m.get("year") else None) not in seen_keys]
    movies.sort(key=lambda m: (m.get("vote_count") or 0), reverse=True)
    return movies[:limit]


# --------------------------------------------------------------------------
# writing it up
# --------------------------------------------------------------------------

def _why_line(movie: dict) -> str:
    """Render the computed overlaps as a short factual clause."""
    why = movie.get("why") or {}
    bits = []
    if why.get("director"):
        bits.append(f"same director ({why['director']})")
    if why.get("genres"):
        bits.append("shared genres: " + ", ".join(why["genres"]))
    if why.get("keywords"):
        bits.append("shared themes: " + ", ".join(why["keywords"]))
    if why.get("cast"):
        bits.append("shared cast: " + ", ".join(why["cast"]))
    return "; ".join(bits) or "no specific overlap computed"


def _rating_label(m: dict) -> str:
    """TMDB's aggregated user rating, or an honest "not enough" note.

    A vote_average built off a handful of votes reads as precise but
    isn't - below a small threshold it's more misleading than useful.
    """
    count = m.get("vote_count") or 0
    if count < 20:
        return "not enough ratings yet"
    return f"{float(m.get('vote_average') or 0):.1f}/10 (TMDB, {count} votes)"


# --------------------------------------------------------------------------
# reranking
# --------------------------------------------------------------------------

RERANK_SYSTEM = """
You are the final filter on a movie recommendation. You are given a
person's request and a numbered shortlist of candidate films that a
ranking system has already judged to be a good match. Your only job is to
choose the best few, in the best order, from that shortlist.

Judge each candidate on the request, its plot summary, and the "match:"
line, which lists the concrete overlaps the ranking system computed. A
film whose plot genuinely fits the request should rank above one that
matches only on paper.

Prefer variety: avoid returning several films from the same series unless
they are clearly the strongest fits.

Rules you must not break:
- Choose only from the numbered candidates. Never name a film that is not
  on the list. Never invent an index.
- Return JSON only, exactly: {"order": [best index, next, ...]}
  listing the indices of your chosen films, best first.
"""


def rerank_picks(candidates: list, user_message: str,
                 reference_titles: list = None, n: int = 3) -> list:
    """Let the model choose the best `n` from an already-good shortlist.

    This is a reranker, not a retriever. Every candidate has already
    cleared the content-similarity floor, so the model only ever chooses
    among films the ranking system already vouched for - it cannot add
    one, and on any failure the ranking system's own order stands. That
    boundary is deliberate: the bug this whole project was built to fix
    was the model inventing a connection to a film nothing had matched,
    and a reranker that can only reorder a vetted shortlist keeps that
    door shut while still applying judgement the cosine blend cannot.
    """
    if not candidates:
        return []
    if len(candidates) <= n:
        return candidates

    lines = []
    for i, m in enumerate(candidates):
        year = str(m.get("release_date") or m.get("year") or "")[:4]
        plot = (m.get("overview") or "")[:200]
        lines.append(f"[{i}] {m['title']} ({year})\n"
                     f"     match: {_why_line(m)}\n"
                     f"     plot: {plot}")

    named = ", ".join(reference_titles or [])
    prompt = (
        f"Request: {user_message!r}\n"
        + (f"They named: {named}\n" if named else "")
        + f"\nChoose the {n} best, best first.\n\nCandidates:\n"
        + "\n".join(lines)
    )

    fallback = candidates[:n]
    raw = ask_ai(prompt, RERANK_SYSTEM, json_mode=True)
    if not raw:
        return fallback
    try:
        order = json.loads(raw).get("order", [])
    except (json.JSONDecodeError, AttributeError):
        log.warning("reranker returned non-JSON: %r", str(raw)[:120])
        return fallback

    # Keep only valid, in-range, non-duplicate indices. Anything the model
    # invents or repeats is silently dropped rather than trusted.
    chosen, seen = [], set()
    for idx in order:
        if not isinstance(idx, int) or idx in seen or not (0 <= idx < len(candidates)):
            continue
        seen.add(idx)
        chosen.append(candidates[idx])
        if len(chosen) >= n:
            break

    if not chosen:
        return fallback

    # If the model returned fewer than n, top up from the blend's order so
    # the list is always full and always drawn from the vetted shortlist.
    for m in candidates:
        if len(chosen) >= n:
            break
        if m not in chosen:
            chosen.append(m)
    return chosen


def format_recommendations(movies: list, original_request: str, explanation: str,
                           taste_context: str = "", reference_titles: list = None,
                           hedge: str = "") -> str:
    reference_titles = reference_titles or []

    movie_list = "\n".join(
        f"- {m['title']} ({str(m.get('release_date') or m.get('year') or '')[:4]}) "
        f"— rating: {_rating_label(m)}\n"
        f"    plot: {(m.get('overview') or '')[:150]}\n"
        f"    why it was matched: {_why_line(m)}"
        for m in movies
    )

    system = """
    You are a passionate, friendly movie recommender on Telegram.
    Write a warm, concise recommendation message. Use a few emojis.
    End with one short sentence encouraging them to watch.
    Do not use markdown bold or italics.

    Formatting rule, followed exactly for every film: a numbered line of
    the exact shape
        N. Title (Year) — rating
    using the rating text given for that film verbatim (including when it
    says there aren't enough ratings yet - never invent or compute a
    number yourself). Put your one-line reason for the pick on the next
    line, indented or dashed, not on the same line as the title.

    Critical rule: the "why it was matched" line for each film is the only
    justification you may use. It was computed by the ranking system. Base
    your one-line reason on it and on the plot summary given.

    Never claim a connection that is not in that line. Do not invent shared
    themes, shared tone, or a shared emotional quality. If a film's overlap
    is thin, say plainly that it is a looser pick rather than overselling
    it. An honest weak recommendation is fine; an invented reason is not.
    """

    reference_note = (
        f"They asked for films like: {', '.join(reference_titles)}. "
        f"You may refer to that film by name."
        if reference_titles else
        "They named no specific film, so do not claim any film inspired these picks."
    )

    prompt = f"""
    User asked for: {original_request!r}
    Your interpretation: {explanation}
    {reference_note}
    {f"Their taste profile: {taste_context}" if taste_context else ""}
    {f"Important caveat you must convey: {hedge}" if hedge else ""}

    Films to recommend:
    {movie_list}
    """

    return ask_ai(prompt, system) or "Here's what I found:\n" + movie_list


def _taste_context(liked_features: list) -> str:
    genres, directors = [], []
    for f in liked_features[:8]:
        for g in f.get("genres", []):
            if g not in genres:
                genres.append(g)
        director = f.get("director")
        if director and director not in directors:
            directors.append(director)

    if not genres:
        return ""
    context = f"Watches a lot of: {', '.join(genres[:4])}."
    if directors:
        context += f" Likes directors: {', '.join(directors[:3])}."
    return context


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def load_profile(username: str, allow_scrape: bool = True,
                 force: bool = False) -> dict | None:
    """Cached Letterboxd profile, scraping only when there isn't a fresh one.

    `force` bypasses the cache for /refresh, which exists because the
    cache TTL is hours and someone who has just logged five films wants
    them counted now.
    """
    if not username:
        return None
    if not force:
        cached = get_cached_profile(username)
        if cached:
            return cached
    if not allow_scrape:
        return None

    profile = get_taste_profile(username)
    if profile:
        put_profile(username, profile)
    return profile


def warm_profile(username: str) -> dict:
    """Do a linked user's slow work up front, right after they link.

    A first recommendation used to pay for everything at once: scrape
    Letterboxd, then one TMDB search plus one enrichment per film, all
    while the user stared at a silent chat and assumed the bot was
    broken. Doing it at link time means the scrape is cached and every
    film is already resolved before they ask for anything.

    Returns a small summary so the caller can tell the user what landed.
    """
    profile = load_profile(username)
    if not profile:
        return {"ok": False, "resolved": 0, "total": 0}

    liked, _ = resolve_films(profile["liked"])
    resolve_films(profile["disliked"][:15])   # cached for the same reason

    return {
        "ok":          True,
        "resolved":    len(liked),
        "total":       len(profile["liked"]),
        "total_films": profile.get("total_films", 0),
        "total_rated": profile.get("total_rated", 0),
        "top_genres":  _top_genres(liked),
    }


def _top_genres(features: list, limit: int = 3) -> list:
    counts = {}
    for f in features:
        for g in f.get("genres", []):
            counts[g] = counts.get(g, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


class Recommendation(NamedTuple):
    """The reply, plus the films it is about.

    The bot needs the films themselves, not just the prose, so it can
    put a thumbs up/down on each one - and a vote is only worth
    collecting if we know which film it refers to.
    """

    text: str
    picks: list = []


def build_profile(username: str = None, user_id: str = None) -> dict | None:
    """Letterboxd history and recommendation votes, merged.

    Either half may be missing. Someone who has never linked an account
    still builds a profile out of their votes alone, which is the point:
    personalisation should not require handing over a Letterboxd login.
    """
    profile = None
    if username:
        try:
            profile = load_profile(username)
        except Exception as exc:
            log.warning("could not load profile for %s: %s", username, exc,
                        exc_info=True)

    fragment = feedback.profile_fragment(user_id) if user_id else None
    return feedback.merge(profile, fragment)


def recommend_message(user_message: str, username: str = None,
                      user_id: str = None) -> Recommendation:
    log.info("processing %r for user %s", user_message, username or user_id)

    vibe = interpret_vibe(user_message)
    if not vibe:
        return Recommendation("My brain's offline for a sec - try me again in a moment!")

    if vibe["intent"] != "recommend":
        return Recommendation(vibe.get("reply") or (
            "Tell me a mood, a genre, or a film you loved, and I'll find you something! "
            "For example: something dark and mind-bending."
        ))

    explanation = vibe["explanation"]
    reference_features = resolve_titles(vibe["reference_titles"])

    named = [f["title"] for f in reference_features]
    unresolved = [t for t in vibe["reference_titles"]
                  if t.lower() not in {n.lower() for n in named}]
    if unresolved:
        log.info("could not resolve named film(s): %s", unresolved)

    # If they named films and we identified none of them, the request is
    # about a film we do not have. Falling through to treat the sentence
    # as a mood would answer a question nobody asked - it is what turned
    # "like Zorblax the Undying" into a list of superhero films.
    if unresolved and not reference_features:
        names = ", ".join(f"\u201c{t}\u201d" for t in unresolved)
        return Recommendation(
            f"I couldn't find {names} in my library, so I'd only be "
            f"guessing at what it's like. Have I got the title right? "
            f"Otherwise give me a mood, or another film you loved, and "
            f"I'll take it from there.")

    profile = build_profile(username, user_id)

    # Retrieve a shortlist, not the final three: the blend is stage one,
    # and the reranker below picks the three shown from this pool. When
    # reranking is off, the pool is just the three itself.
    pool_n = config.RERANK_POOL if config.RERANK_ENABLED else config.RECOMMEND_N
    try:
        pool = recommend(
            profile=profile,
            reference_features=reference_features,
            keywords=vibe["keywords"],
            # The raw message, not just the extracted keywords: the
            # semantic signal reads "dark and mind-bending" directly,
            # which is the whole reason a vibe request works at all now.
            query_text=user_message,
            genre_filter=vibe["genres"],
            top_n=pool_n,
        )
    except Exception as exc:
        log.warning("ranking failed: %s", exc, exc_info=True)
        pool = []

    if config.RERANK_ENABLED and len(pool) > config.RECOMMEND_N:
        try:
            picks = rerank_picks(pool, user_message, named, n=config.RECOMMEND_N)
        except Exception as exc:
            log.warning("rerank failed, using blend order: %s", exc)
            picks = pool[:config.RECOMMEND_N]
    else:
        picks = pool[:config.RECOMMEND_N]

    # Taste context describes the *user*, so it has to come from their own
    # liked films - describing the picks back to them would be circular,
    # and describing the film they named is not a taste profile at all.
    # These lookups are already cached by the ranking pass above.
    taste_context = ""
    if profile and picks:
        try:
            liked_features, _ = resolve_films(profile["liked"][:8])
            taste_context = _taste_context(liked_features)
        except Exception as exc:
            log.debug("could not build taste context: %s", exc)

    if picks:
        return Recommendation(
            format_recommendations(picks, user_message, explanation,
                                   taste_context, named), picks)

    # Nothing in the corpus cleared the score floor. Widen rather than
    # lower the bar: ask TMDB directly about the film they named.
    if reference_features:
        seen_keys = (profile or {}).get("watched_keys", set())
        neighbours = fetch_similar_to(reference_features, seen_keys)
        if neighbours:
            return Recommendation(
                format_recommendations(
                    neighbours, user_message, explanation, "", named,
                    hedge="These come from TMDB's own similar-films list for that "
                          "title rather than a close match in my own library."),
                neighbours)

    movies = fetch_movies_by_vibe(vibe)
    if not movies:
        return Recommendation(
            "I couldn't find a good match for that one - try naming a film "
            "you loved, or describing the vibe a different way!")

    return Recommendation(
        format_recommendations(
            movies, user_message, explanation, "", [],
            hedge="Nothing scored as a strong match, so these are well-rated films "
                  "in the right genre rather than close matches. Say so clearly, "
                  "and do not invent a connection to anything they mentioned."),
        movies)


def get_vibe_recommendations(user_message: str, username: str = None,
                             user_id: str = None) -> str:
    """Text-only wrapper, for callers that don't need the films."""
    return recommend_message(user_message, username, user_id).text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(get_vibe_recommendations("something like Interstellar"))
