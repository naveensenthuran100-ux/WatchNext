import requests
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
from src.tmdb import tmdb_get, search_movie
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

PRIMARY_MODEL  = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "groq/compound-mini"

def ask_ai(prompt, system):
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages= [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, 
                max_tokens=500
            )
            return response.choices[0].message.content
        
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str (e):
                print(f"[fallback] {model} rate limited, trying next..")
                continue
            else:
                raise e
    return None

def interpret_vibe(user_message):
    system = """
    You are a movie expert assistant.
    A user will describe a movie vibe or mood in natural language.
    Extract structured information from their message.

    Reply ONLY in this exact format, nothing else:
    GENRES: genre1, genre2
    MOOD: one word
    KEYWORDS: keyword1, keyword2, keyword3
    DECADE: any/80s/90s/00s/10s/20s
    EXPLANATION: one sentence explaining your interpretation
    """

    prompt = f"The user wants: '{user_message}'. Extract the movie parameters."

    return ask_ai(prompt, system)

def parse_vibe_response(ai_response):
    result = {}
    for line in ai_response.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result
    
def get_genre_ids(genre_names):
    data = tmdb_get("/genre/movie/list")
    all_genres = data.get("genres", [])

    genre_map = {}
    for g in all_genres:
        name = g["name"].lower()
        id   = g["id"]
        genre_map[name] = id

    ids = []
    for name in genre_names:
        name = name.strip().lower()
        if name in genre_map:
            ids.append(str(genre_map[name]))
    
    return ids

def fetch_movies_by_vibe(parsed_vibe):
    genre_names = parsed_vibe.get("GENRES", "").split(",")
    decade      = parsed_vibe.get("DECADE", "any")

    genre_ids = get_genre_ids(genre_names)

    params = {
        "sort_by":        "vote_average.desc",
        "vote_count.gte": 100,
        "with_genres":    ",".join(genre_ids) if genre_ids else None,
    }

    decade_ranges = {
        "80s": ("1980-01-01", "1989-12-31"),
        "90s": ("1990-01-01", "1999-12-31"),
        "00s": ("2000-01-01", "2009-12-31"),
        "10s": ("2010-01-01", "2019-12-31"),
        "20s": ("2020-01-01", "2029-12-31"),
    }

    if decade in decade_ranges:
        start, end = decade_ranges[decade]
        params["primary_release_date.gte"] = start
        params["primary_release_date.lte"] = end

    params = {k: v for k, v in params.items() if v is not None}

    data = tmdb_get("/discover/movie", params=params)

    return data.get("results", [])[:8]

def format_recommendations(movies, original_request, explanation, taste_context: str = ""):
    movie_list = "\n".join([
        f"- {m['title']} ({m.get('release_date', '')[:4]}) "
        f"🤩{m['vote_average']:.1f} — {m['overview'][:80]}..."
        for m in movies
    ])

    system = """
    You are a passionate, friendly movie recommender on Telegram.
    Given a list of movies and what the user asked for, write a warm
    engaging recommendation message. Use emojis. Keep it concise.
    Format each movie as a numbered list with a one line reason why it fits.
    If taste context is provided, naturally mention 1-2 specific overlaps
    (e.g. "leans into your sci-fi streak" or "same director DNA as your favourites").
    End with one short hype sentence encouraging them to watch.
    """

    prompt = f"""
    User asked for: '{original_request}'
    Your interpretation: {explanation}
    {f"Their taste profile: {taste_context}" if taste_context else ""}

    Movies to recommend:
    {movie_list}

    Write the Telegram recommendation message now.
    """

    return ask_ai(prompt, system)

def get_vibe_recommendations(user_message: str, username: str = None) -> str:
    print(f"[recommender] Processing: '{user_message}' for user: {username}")

    ai_response = interpret_vibe(user_message)
    if not ai_response:
        return "Sorry, taking a smoke break, gimme 5min!"

    print(f"[recommender] AI interpreted: {ai_response}")
    parsed      = parse_vibe_response(ai_response)
    explanation = parsed.get("EXPLANATION", "")

    if not username:
        movies = fetch_movies_by_vibe(parsed)
        if not movies:
            return "cant get your vibe, tell me your vibe in a diff way!"
        return format_recommendations(movies, user_message, explanation)

    from src.taste import recommend_for_letterboxd_user, build_soup, fit_vectorizer, build_taste_vector, rank_candidates, get_candidate_pool, clean_title
    from src.tmdb import search_movie
    from src.cache import cached_features
    from src.letterboxd import get_taste_profile
    import re

    profile = get_taste_profile(username)
    if not profile:
        movies = fetch_movies_by_vibe(parsed)
        return format_recommendations(movies, user_message, explanation)

    all_titles = []
    seen = set()
    for title in profile["recent_films"] + profile["highly_rated"]:
        if title not in seen:
            all_titles.append(title)
            seen.add(title)

    liked_features = []
    star_ratings   = []
    for title in all_titles[:20]:
        result = search_movie(clean_title(title))
        if result:
            features = cached_features(result["id"])
            liked_features.append(features)
            star_ratings.append(4.0)

    if not liked_features:
        movies = fetch_movies_by_vibe(parsed)
        return format_recommendations(movies, user_message, explanation)

    vibe_genres  = [g.strip().lower() for g in parsed.get("GENRES", "").split(",")]
    pool         = get_candidate_pool(liked_features)

    vibe_filtered = [
        f for f in pool
        if any(g.lower() in vibe_genres for g in f["genres"])
    ]

    if len(vibe_filtered) < 10:
        vibe_filtered = pool

    liked_soups     = [build_soup(f) for f in liked_features]
    candidate_soups = [build_soup(f) for f in vibe_filtered]
    all_soups       = liked_soups + candidate_soups

    vectorizer, all_matrix = fit_vectorizer(all_soups)
    liked_matrix     = all_matrix[:len(liked_soups)]
    candidate_matrix = all_matrix[len(liked_soups):]

    taste_vector = build_taste_vector(liked_matrix, star_ratings)
    ranked       = rank_candidates(taste_vector, candidate_matrix, vibe_filtered)

    watched = set(f["title"].lower().strip() for f in liked_features)
    unseen  = [f for f in ranked if f["title"].lower().strip() not in watched]

    top_films = []
    for f in unseen[:8]:
        top_films.append({
            "title":        f["title"],
            "release_date": f["year"],
            "vote_average": f["vote_average"],
            "overview":     f["overview"]
        })

    if not top_films:
        movies = fetch_movies_by_vibe(parsed)
        return format_recommendations(movies, user_message, explanation)

    all_genres   = []
    all_directors = []
    for f in liked_features[:5]:  
        for g in f["genres"]:
            if g not in all_genres:
                all_genres.append(g)
        if f["director"] and f["director"] not in all_directors:
            all_directors.append(f["director"])

    taste_context = f"Tends to watch: {', '.join(all_genres[:4])}. "
    if all_directors:
        taste_context += f"Likes directors: {', '.join(all_directors[:3])}."

    return format_recommendations(top_films, user_message, explanation, taste_context)
if __name__ == "__main__":
    test = get_vibe_recommendations("something sad,and makes you cry, but makes you feel alive after ")
    print(test)