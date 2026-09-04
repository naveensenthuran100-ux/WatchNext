import requests
from bs4 import BeautifulSoup

BASE_URL = "https://letterboxd.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

def get_recent_films(username):
    url = f"{BASE_URL}/{username}/films/"
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    film_containers = soup.find_all("li", class_="griditem")
    
    films = []
    for container in film_containers:
        poster = container.find("div", class_="react-component")
        if poster:
            title = poster.get("data-item-name", "")  # ← was data-film-slug
            films.append(title)                        # ← removed .replace and .title()
    
    return films[:20]

def get_rated_films(username):
    url = f"{BASE_URL}/{username}/films/"  # ← same page as recent films!
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return {}
    
    soup = BeautifulSoup(response.text, "html.parser")
    film_containers = soup.find_all("li", class_="griditem")
    
    rated_films = {}
    for container in film_containers:
        poster = container.find("div", class_="react-component")
        rating = container.find("span", class_="rating")
        
        if poster and rating:
            title = poster.get("data-item-name", "")
            stars = rating.text.strip().count("★")
            rated_films[title] = stars
    
    return rated_films


def get_taste_profile(username):
    recent = get_recent_films(username)
    rated  = get_rated_films(username)
    
    if not recent and not rated:
        return None
    
    highly_rated = [
        title for title, stars in rated.items()
        if stars >= 4
    ]
    
    profile = {
        "username":     username,
        "recent_films": recent[:10],
        "highly_rated": highly_rated[:10],
        "total_rated":  len(rated)
    }
    
    return profile

if __name__ == "__main__":
    username = input("Enter a Letterboxd username: ")
    profile  = get_taste_profile(username)
    
    if profile:
        print(f"\nUsername: {profile['username']}")
        print(f"Recent films: {profile['recent_films']}")
        print(f"Highly rated: {profile['highly_rated']}")
        print(f"Total rated:  {profile['total_rated']}")
    else:
        print("Couldn't fetch profile — check the username!")