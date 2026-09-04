# 🎬 Letterboxd Movie Recommender Bot

A Telegram bot that gives personalised movie recommendations based on your Letterboxd taste profile. Describe a vibe in plain English — it figures out what you mean and finds films that actually match both the mood *and* your watching history.

---

## What it does

Most recommendation systems either ignore your taste (generic "top rated" lists) or ignore your current mood (just "more of what you watched"). This bot does both.

- **Vibe parsing** — you say something like *"something dark and mind-bending"* or *"a feel-good film for a Sunday"*, and an LLM interprets that into genre, mood, keywords, and decade signals
- **Taste modelling** — it reads your Letterboxd profile (recent watches + highly rated films) and builds a feature vector of your preferences
- **Personalised ranking** — candidates from TMDB are scored against your taste vector using cosine similarity, then filtered to match the vibe you described
- **Feature caching** — TMDB lookups are cached locally so repeat queries are fast and don't burn API quota

---

## How it works

```
User message → LLM vibe parsing → TMDB candidate pool
                                         ↓
Letterboxd profile → TF-IDF taste vector → cosine similarity ranking
                                         ↓
                              Top unseen matches → LLM formats response
```

### The recommendation pipeline

**1. Vibe interpretation (`recommender.py`)**
The user's message is sent to a Groq-hosted LLM which extracts structured parameters: genres, mood, keywords, and decade. This handles the messy natural language part so the retrieval step works cleanly.

**2. Feature extraction (`tmdb.py`, `cache.py`)**
For each film, the bot fetches genres, keywords, top 3 cast members, director, and overview from TMDB. These are cached in `data/features_cache.json` so the same film is never fetched twice.

**3. Taste vector construction (`taste.py`)**
Each film's features are turned into a "soup" string — a weighted concatenation of tokens where genres and director are repeated 3x (they're stronger taste signals than cast or keywords). These soups are vectorised with TF-IDF across the full corpus, and the user's liked films are averaged into a single taste vector, weighted by star rating.

**4. Candidate pool + ranking (`taste.py`)**
Candidates come from two sources: TMDB Discover filtered by the user's genre preferences, and TMDB's own recommendation endpoint seeded from their top liked films. Candidates are then scored by cosine similarity against the taste vector and sorted.

**5. Vibe filtering**
The ranked candidates are filtered to only those matching the genres from the vibe parse. If that leaves fewer than 10 films, the filter is relaxed and falls back to the full ranked pool.

**6. Response formatting (`recommender.py`)**
The top 8 unseen films are passed back to the LLM with taste context (dominant genres, favourite directors) to write a warm, personalised Telegram message.

---

## Project structure

```
.
├── bot.py              # Telegram bot — command handlers and message routing
├── recommender.py      # Main pipeline: vibe parsing, retrieval, formatting
├── taste.py            # TF-IDF vectoriser, taste vector, candidate ranking
├── letterboxd.py       # Letterboxd scraper (recent films + ratings)
├── tmdb.py             # TMDB API wrapper
├── cache.py            # Disk-backed feature cache
├── evaluate.py         # Leave-one-out evaluation harness
├── data/
│   └── features_cache.json   # Cached TMDB features (gitignored)
├── requirements.txt
└── .env                # API keys (gitignored)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/naveensenthuran100-ux/WatchNext.git
cd letterboxd-bot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your API keys

| Key | Where to get it |
|-----|----------------|
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) on Telegram — `/newbot` |
| `TMDB_TOKEN` | [themoviedb.org](https://www.themoviedb.org/settings/api) → API → Bearer token |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

### 3. Create your `.env`

```
BOT_TOKEN=your_telegram_bot_token
TMDB_TOKEN=your_tmdb_bearer_token
GROQ_API_KEY=your_groq_api_key
```

### 4. Run

```bash
python bot.py
```

---

## Usage

| Command | Description |
|---------|-------------|
| `/start` | Introduction and usage guide |
| `/setletterboxd <username>` | Link your Letterboxd account |
| `/rec <vibe>` | Get personalised recommendations |
| *(any message)* | Treated as a vibe — same as `/rec` |

**Example session:**
```
/setletterboxd dave
/rec something dark and cerebral, like a fever dream
```

Without a linked Letterboxd account, recommendations still work — they're just not personalised to your taste.

---

## Evaluation

`evaluate.py` runs a leave-one-out test: for each film in your Letterboxd history, it hides that film, builds a taste vector from the rest, and checks what rank the hidden film gets in the candidate pool.

```bash
python evaluate.py
```

Metrics reported:
- **hit@10** — fraction of films that rank in the top 10 when hidden
- **median rank** — middle rank across all tested films

---

## Limitations worth knowing

- **Letterboxd scraping** — the scraper reads the HTML directly, so it can break if Letterboxd changes their markup. It currently reads from the `/films/` page, which only shows recent/rated films — not full diary history.
- **Star ratings default to 4.0** — when building the taste vector from Letterboxd, all films are currently weighted equally at 4 stars. Actual star ratings from Letterboxd are read where available but the pipeline defaults to 4.0 across the board. A proper weighted taste vector using real ratings is an obvious next improvement.
- **Cold start** — if the Letterboxd profile has fewer than ~5 films, the taste vector is too thin to be meaningful and the bot falls back to vibe-only recommendations.
- **Cache grows unbounded** — `features_cache.json` accumulates every film ever looked up. Fine for personal use; worth pruning or switching to SQLite if you scale up.

---

## Tech stack

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) — bot framework
- [TMDB API](https://developer.themoviedb.org) — film metadata
- [Groq](https://console.groq.com) — LLM inference for vibe parsing and response formatting
- [scikit-learn](https://scikit-learn.org) — TF-IDF vectorisation and cosine similarity
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) — Letterboxd scraping