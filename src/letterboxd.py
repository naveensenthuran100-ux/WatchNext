"""Letterboxd profile scraper.

Letterboxd has no public API, so this parses the /films/ grid. Notes on
what the markup gives us:

  data-item-name  ->  "Dune (2021)"  (title AND year, so we can
                      disambiguate the TMDB search instead of guessing)
  class="rated-8" ->  rating on a 0-10 scale, i.e. 4 stars. Counting the
                      "*" glyphs loses half-stars, collapsing 4.5 into 4.

The original fetched the same URL twice (once for titles, once for
ratings). This fetches each page once and parses everything out of it.

Verified limitation: Letterboxd returns 403 for /films/page/2/ and beyond
regardless of user-agent, session cookies or delay, so one page (72 films)
is the practical ceiling per profile. max_pages is kept for the day that
changes, but defaults to 1 so we don't spend a blocked request each time.
"""
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from src import config

log = logging.getLogger(__name__)

BASE_URL = "https://letterboxd.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# "Dune (2021)" -> ("Dune", "2021")
_TITLE_YEAR = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def clean_title(name: str) -> str:
    match = _TITLE_YEAR.match(name.strip())
    return match.group(1).strip() if match else name.strip()


def split_title_year(name: str) -> tuple[str, str | None]:
    match = _TITLE_YEAR.match(name.strip())
    if match:
        return match.group(1).strip(), match.group(2)
    return name.strip(), None


def _parse_rating(container) -> float | None:
    """Read the exact rating from the rated-N class (N is out of 10)."""
    span = container.find("span", class_="rating")
    if not span:
        return None
    for css_class in span.get("class", []):
        if css_class.startswith("rated-"):
            try:
                return int(css_class.split("-")[1]) / 2.0
            except (ValueError, IndexError):
                pass
    text = span.get_text(strip=True)          # fallback if the class changes
    return text.count("★") + 0.5 * text.count("½") or None


def _parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    films = []
    for container in soup.find_all("li", class_="griditem"):
        poster = container.find("div", class_="react-component")
        if not poster:
            continue
        name = poster.get("data-item-name") or poster.get("data-item-full-display-name") or ""
        if not name:
            continue
        title, year = split_title_year(name)
        films.append({
            "title":  title,
            "year":   year,
            "slug":   poster.get("data-item-slug", ""),
            "rating": _parse_rating(container),
        })
    return films


def get_films(username: str, max_pages: int = 1) -> list[dict]:
    """Scrape a user's film grid, newest first, across up to max_pages."""
    films = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/{username}/films/"
        if page > 1:
            url += f"page/{page}/"

        try:
            response = requests.get(url, headers=HEADERS, timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("letterboxd request failed for %s page %s: %s", username, page, exc)
            break

        if response.status_code == 404:
            log.info("letterboxd user %r not found", username)
            break
        if response.status_code != 200:
            log.warning("letterboxd returned %s for %s", response.status_code, url)
            break

        page_films = _parse_page(response.text)
        if not page_films:
            break

        films.extend(page_films)
        if len(page_films) < 72:      # short page means we hit the end
            break
        if page < max_pages:
            time.sleep(0.5)           # be polite; this is a scrape, not an API

    return films


def get_taste_profile(username: str, max_pages: int = 1) -> dict | None:
    """Build a taste profile.

    Every film watched is returned (used to avoid recommending something
    already seen), but only well-rated ones drive the taste vector.
    """
    films = get_films(username, max_pages=max_pages)
    if not films:
        return None

    rated = [f for f in films if f["rating"] is not None]

    liked = sorted(
        (f for f in rated if f["rating"] >= config.LIKE_THRESHOLD),
        key=lambda f: f["rating"],
        reverse=True,
    )
    disliked = [f for f in rated if f["rating"] <= config.DISLIKE_THRESHOLD]

    # Recent unrated films are a weak positive signal - somebody chose to
    # watch them - so they backfill when a user rates very little.
    if len(liked) < 5:
        liked = liked + [f for f in films if f["rating"] is None][:10]

    return {
        "username":    username,
        "films":       films,
        "liked":       liked[:config.MAX_LIKED_FILMS],
        "disliked":    disliked[:config.MAX_LIKED_FILMS],
        "watched_keys": {
            (f["title"].lower().strip(), f["year"]) for f in films
        },
        "total_films": len(films),
        "total_rated": len(rated),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = input("Enter a Letterboxd username: ")
    profile = get_taste_profile(name)
    if profile:
        print(f"\nUsername:    {profile['username']}")
        print(f"Films seen:  {profile['total_films']} ({profile['total_rated']} rated)")
        print(f"Liked:       {[f['title'] for f in profile['liked'][:10]]}")
        print(f"Disliked:    {[f['title'] for f in profile['disliked'][:5]]}")
    else:
        print("Couldn't fetch profile - check the username!")
