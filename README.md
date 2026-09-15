# 🎬 WatchNext — a Letterboxd-aware movie recommendation bot

A Telegram bot that recommends films. Name a film you loved, describe a
mood, or link your Letterboxd account — it ranks a local corpus of ~5,400
films against what you actually asked for, and explains each pick using
overlaps it computed rather than ones it made up.

---

## How it decides

Three separate questions, three separate signals. Blending them is the
whole design: no single one of them is good at all three jobs.

| Signal | What it knows | Where it wins |
|---|---|---|
| **metadata** | TF-IDF over genres, keywords, director, cast, franchise | exact shared facts — same director, same series |
| **semantic** | dense embeddings of the film's synopsis and themes | free-text requests: "dark and mind-bending" |
| **collab** | MovieLens item factors — who liked what | tone and feel that appear in no metadata field |
| **your votes** | 👍/👎 you gave on previous recommendations | you, specifically — and it improves with use |

A request produces up to three query vectors, which are then combined:

```
"something like Interstellar"  ──►  reference vector (the named film)
"dark and mind-bending"        ──►  query vector (the raw sentence)
your Letterboxd history        ──►  taste vector (who you are)

        what you asked for leads · taste breaks ties
                        │
                  blended score
                        │
        filters: unseen · mode · genre · z-score floor
                        │
        retrieve a shortlist ─► LLM reranks to the final 3
                        │
              explanation from real overlaps
```

The last two stages are the standard "retrieve cheaply, rerank
precisely" split. The blend is a bi-encoder — it scores each film against
the request independently — so it is fast over 13k films but blind to how
a candidate reads *next to* the request. A second pass hands the LLM the
shortlist it produced and lets it pick the three shown. Measured on the
benchmark, this left named-film precision@3 unchanged (0.42) and lifted
free-text precision@3 from 0.19 to 0.26. Crucially the reranker can only
reorder and trim a shortlist that already cleared the similarity floor —
it can never introduce a film, and any failure falls straight back to the
blend's own order. That boundary is what keeps it from reviving the
original bug, where the model invented a connection to a film nothing had
matched. Set `RERANK_ENABLED=0` to skip it.

### Measured

`python -m src.bench` scores each signal against the corpus of **13,092
films**. hit@10 (a film a human would obviously accept landing in the top
ten) and MRR are shown because, unlike raw median rank, they stay
comparable as the corpus grows:

| signal | "like X" hit@10 / MRR | free-text hit@10 / MRR |
|---|---|---|
| metadata | 0.233 / 0.128 | 0.220 / 0.104 |
| semantic | 0.083 / 0.033 | 0.244 / 0.130 |
| collab † | 0.129 / 0.103 | — |
| **blend** | **0.250 / 0.147** | **0.268 / 0.125** |

† over the 31% of films MovieLens covers on this corpus. Each signal wins
a different job — metadata on named films, semantic on free text — and
the per-context blend (see below) beats every single signal on both.
Collab has no free-text vector: it only knows films it has ratings for.

### What did not work

Three plausible ideas were built, measured, and rejected. They are kept
in the tree because the measurement is the useful part.

- **A transformer encoder** (`bge-small` via ONNX) in place of static
  embeddings. Re-tested head-to-head on the 13k corpus: bge is *better* at
  film-to-film (reference MRR 0.077 vs 0.033) and *worse* at free-text
  (query MRR 0.047 vs 0.130) — the opposite of the reason to want it,
  since free text is the whole point of the semantic signal. It also
  costs ~10 min to encode the corpus against static's ~2 s. Static
  embeddings suit these documents because the documents are field dumps,
  not prose, and short mood queries match them better token-by-token than
  a contextual encoder does. Kept static (`potion-base-8M`).
- **TMDB's recommendation graph** (`src/graph.py`) as a second
  collaborative signal. It covers 100% of the corpus where MovieLens
  covers 46%, and scores well alone (reference median 60). But it is
  twice as correlated with the content signals as MovieLens is (Spearman
  0.35 vs 0.19), so it largely re-derives what metadata already knows.
  Adding it improved the reference median while costing top-10 hits, and
  a paired test could not separate it from noise (p = 0.39). It is built,
  cached and weighted at zero; set `W_GRAPH` to bring it back.
- **Reciprocal Rank Fusion** in place of the standardised-score blend.
  RRF combines each signal's *ranking* rather than its scores, which is
  scale-free by construction. Measured, it split exactly the way the two
  contexts do: it lifted free-text top-10 recall (0.268 → 0.366) but
  flattened the sharp metadata signal that "like X" leans on and cost
  reference MRR (0.147 → 0.109). A win in one job and a loss in the other
  is not worth a second fusion path, so the z-score blend stays.
- **Replacing the centroid with top-K** outright. See below.

The lesson worth keeping: what earns a signal a place in a blend is its
*independence* from the others, not its standalone accuracy.

### Two design choices that came out of the harness rather than intuition

- **Scores are standardised before blending.** A metadata cosine runs
  0.005–0.16 and a semantic one runs 0.22–0.66. Averaging them raw let
  semantic's *baseline* outweigh metadata's entire range, and the blend
  scored worse than either signal alone.
- **The user model is half centroid, half top-K.** A centroid is better
  for coherent taste; top-K is better when someone likes two unrelated
  things and the average of horror and rom-coms points at neither. Half
  and half measured best on both.
- **The blend uses different signal weights per context.** A named film
  and a typed mood are answered from different signals, so one weight
  vector was a bad compromise. When a film is named, the precise metadata
  facts lead (`SIGNAL_WEIGHTS`, metadata/semantic/collab 0.40/0.40/0.20);
  for free text, semantic dominates (`QUERY_SIGNAL_WEIGHTS`, 0.25/0.75,
  and collab has no query vector at all). Grid-searched against the
  benchmark, this lifted reference MRR 0.133 → 0.147 and free-text MRR
  0.118 → 0.127 over a single shared vector — the old vector had weighted
  metadata *least* in the very context it is strongest.

Thresholds are in standard deviations, not raw similarity, so the bar for
"this is a real match" keeps meaning the same thing when the signals
underneath it change.

### Learning from you

Every recommendation carries 👍/👎 buttons. A vote does two things: the
film joins your taste profile at the extremes of the rating scale, and it
is marked seen so it never comes back. A thumbs-down sticks.

This works with no Letterboxd account at all — three 👍 on Pixar films is
enough to start returning Pixar. It is also the only signal here that is
*about you rather than about films*, and by construction independent of
the other three: a film you rejected is one the content signals had
already ranked highly, so your verdict carries information none of them
had.

A named film still leads the ranking (`REFERENCE_WEIGHT`, 0.75) because
naming one is unambiguous. A mood does not, so there taste gets an equal
say (`QUERY_WEIGHT`, 0.5) — a judgement call, not a measured one.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```
BOT_TOKEN=your_telegram_bot_token
TMDB_TOKEN=your_tmdb_bearer_token
GROQ_API_KEY=your_groq_api_key
```

| Key | Where |
|---|---|
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) → `/newbot` |
| `TMDB_TOKEN` | [themoviedb.org](https://www.themoviedb.org/settings/api) → API → Bearer token |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

### Build the model

```bash
python -m src.train      # fetch the film corpus, fit TF-IDF   (~10 min)
python -m src.embed      # embed the corpus                    (~3 s)
python -m src.collab     # MovieLens item factors              (~5 min, ~400MB download)
python -m src.bot        # run it
```

Optionally, `python -m src.graph` builds the TMDB recommendation graph
(~7 min). It is disabled by default — see "What did not work".

The official MovieLens download (`files.grouplens.org`) has had an expired
TLS certificate since 2026-08-28, so `src.collab` falls back to a mirror
of MovieLens-20M. Verification is never disabled. To check, and to switch
over once GroupLens renews:

```bash
python -m src.collab --check      # which dataset, is the official one back?
python -m src.collab --upgrade    # switch to official if it is reachable
```

`src.embed` and `src.collab` are optional — the bot runs on whatever
signals exist and says which are active in `/status`. Re-run `src.embed`
after `src.train`, since the corpus it embeds will have changed.

---

## Commands

| Command | Description |
|---|---|
| `/start` | Introduction |
| `/setletterboxd <username>` | Link a Letterboxd account |
| `/rec <vibe>` | Recommendations — name a film or describe a mood |
| `/status` | Library size, which signals are active, whether your profile is warm |
| `/refresh` | Re-read your Letterboxd after logging new films |
| 👍 / 👎 buttons | Rate any recommendation — teaches the bot, no account needed |
| `/whoami` | Show the linked account |
| *(any message)* | Treated the same as `/rec` |

Linking takes 30–60 seconds the first time (a Letterboxd scrape plus one
TMDB lookup per film). The bot says so up front, keeps the typing
indicator alive, does the heavy resolution in the background, and
messages you when it finishes — the wait was there before, it just
looked like a hang.

---

## Layout

```
src/
  bot.py          Telegram handlers, latency UX, rate limiting
  recommender.py  request → intent → ranking → grounded write-up
  taste.py        the ranking entry point, filters, explanations
  signals.py      Signal interface + metadata and semantic signals
  collab.py       MovieLens download, mapping, item factors
  scoring.py      blending, score standardisation
  corpus.py       the canonical film ordering every signal aligns to
  bench.py        A/B harness: reference, query, cohort, mixed, profile
  train.py        corpus discovery, TF-IDF fitting, weight tuning
  embed.py        build corpus embeddings
  graph.py        TMDB recommendation graph (built, benched, off by default)
  cache.py        SQLite feature/search/profile cache, schema-versioned
  tmdb.py         TMDB client, scored title matching, feature extraction
  letterboxd.py   profile scraper
  evaluate.py     older metadata-only LOO harness (superseded by bench.py)
data/             movies.db, model.joblib, embeddings.npz, collab.npz
tests/            pytest suite
```

---

## Known limits

- **Letterboxd gives 72 films.** `/films/page/2/` returns 403 regardless
  of headers or delay, so one page is the ceiling per profile. A CSV
  import (Letterboxd → Settings → Import & Export) would lift this and is
  the single biggest available improvement to personalisation.
- **Collaborative coverage is 46%.** The MovieLens mirror in use is
  2015-vintage, so films newer than that get no behavioural signal and
  fall back to content. Coverage is masked per film, not faked.
- **Static embeddings are literal-ish.** They are a bag of token vectors,
  so "dark and mind-bending" partly matches films with "dark" in the
  title. A transformer encoder would read the phrase properly, at the
  cost of a ~2GB torch dependency.
- **Personalisation is the weakest path.** Leave-one-out on a real
  profile remains poor (median ~930 of 5,423). Some of that is a genuinely
  hard problem, some is the 72-film ceiling, and some is that a profile of
  very recent releases gets no collaborative signal at all.
- **The feedback loop needs use to pay off.** Votes work from the first
  one, but a handful of ratings will not outweigh the content signals.
  This is the one part of the system that cannot be improved by a better
  benchmark run — only by people using it.

---

## Tests

```bash
python -m pytest tests/ -q
```

Every test corresponds to a real failure — a named film that was never
looked up, a TV series bound to an unrelated documentary, a making-of
ranked as a similar film, a list padded out with films nobody asked for.
