"""Telegram front end.

Three things matter here beyond the handlers themselves:

1. get_vibe_recommendations is fully synchronous (requests + Groq +
   sklearn) and can run for tens of seconds. Awaiting it directly on the
   event loop froze the whole bot for every other user, so it runs in a
   worker thread.
2. Telegram rejects the entire message if Markdown is malformed, and LLM
   prose contains stray * and _ constantly. We send plain text, and only
   retry without formatting if a parse error somehow occurs.
3. Linking a Letterboxd account is slow - a scrape of a third-party page,
   then a TMDB lookup per film. Silence during that reads as a broken
   bot, so every slow step announces itself up front, keeps the typing
   indicator alive while it works, and the heavy resolution happens in
   the background right after linking rather than inside the user's
   first request.
"""
import asyncio
import json
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src import cache, config, feedback
from src.recommender import load_profile, recommend_message, warm_profile
from src.scoring import load_default

log = logging.getLogger(__name__)


def _load_users() -> dict:
    if config.USERS_FILE.exists():
        try:
            return json.loads(config.USERS_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read %s: %s", config.USERS_FILE, exc)
    return {}


user_profiles = _load_users()

# Usernames whose background warm-up is still running, so a request that
# arrives mid-warm can say so instead of looking hung.
_warming: set[str] = set()

# One in-flight request per user. Two overlapping requests would each do
# the same uncached TMDB work and race each other into the same cache.
_locks: dict[str, asyncio.Lock] = {}

# Crude per-user rate limit: this fans out to a paid LLM and a rate-limited
# API, and a held-down send button should not be able to spend either.
_last_request: dict[str, float] = {}
MIN_SECONDS_BETWEEN = 3.0


def _save_users() -> None:
    tmp = config.USERS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(user_profiles, indent=2))
    tmp.replace(config.USERS_FILE)          # atomic; never a half-written file


async def _send(update: Update, text: str, markup=None) -> None:
    try:
        await update.message.reply_text(text, reply_markup=markup)
    except BadRequest as exc:
        log.warning("send failed (%s) - retrying truncated", exc)
        await update.message.reply_text(text[:4000], reply_markup=markup)


# Telegram caps callback_data at 64 bytes, so votes are encoded as
# "v:<+1|-1>:<tmdb id>" and nothing else - the film is looked up rather
# than carried around in the button.
VOTE_PREFIX = "v"


def _vote_keyboard(picks: list) -> InlineKeyboardMarkup | None:
    """A thumbs up/down pair per recommended film.

    Labelled with the title rather than a number: the numbering in the
    prose comes from the model and cannot be relied on to match.
    """
    rows = []
    for film in picks:
        movie_id = film.get("id")
        if movie_id is None:
            continue
        title = str(film.get("title") or "")[:24]
        rows.append([
            InlineKeyboardButton(f"👍 {title}", callback_data=f"{VOTE_PREFIX}:1:{movie_id}"),
            InlineKeyboardButton("👎", callback_data=f"{VOTE_PREFIX}:-1:{movie_id}"),
        ])
    return InlineKeyboardMarkup(rows) if rows else None


async def on_vote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record a thumbs up/down, acknowledge it, and remove that film's buttons.

    Leaving the buttons up after a tap reads as "nothing happened" - the
    only feedback was a small toast that's easy to miss. Pulling the row
    once it's voted is the visible confirmation.
    """
    query = update.callback_query
    try:
        _, verdict, movie_id_str = query.data.split(":")
        verdict, movie_id = int(verdict), int(movie_id_str)
    except (AttributeError, ValueError):
        await query.answer("Couldn't read that one, sorry!")
        return

    user_id = str(query.from_user.id)
    await asyncio.to_thread(feedback.record, user_id, movie_id, verdict)

    features = await asyncio.to_thread(cache.get_cached, movie_id)
    title = (features or {}).get("title", "that one")
    up, down = await asyncio.to_thread(cache.feedback_counts, user_id)

    if verdict > 0:
        note = f"Noted — more like {title}. ({up} 👍, {down} 👎)"
    else:
        note = f"Got it — less like {title}. ({up} 👍, {down} 👎)"
    await query.answer(note)

    markup = query.message.reply_markup if query.message else None
    if markup:
        remaining = [
            row for row in markup.inline_keyboard
            if not any(btn.callback_data and
                      btn.callback_data.endswith(f":{movie_id_str}")
                      for btn in row)
        ]
        if len(remaining) != len(markup.inline_keyboard):
            new_markup = InlineKeyboardMarkup(remaining) if remaining else None
            try:
                await query.edit_message_reply_markup(reply_markup=new_markup)
            except BadRequest as exc:
                log.debug("could not update vote buttons: %s", exc)


class _Typing:
    """Keep the 'typing...' indicator alive for a long operation.

    Telegram expires a chat action after about five seconds. A single
    send_action at the start of a thirty-second job leaves the user
    looking at a still chat for twenty-five of them.
    """

    def __init__(self, chat, interval: float = 4.0):
        self._chat = chat
        self._interval = interval
        self._task = None

    async def _loop(self):
        try:
            while True:
                try:
                    await self._chat.send_action(ChatAction.TYPING)
                except Exception as exc:          # never kill the request
                    log.debug("typing indicator failed: %s", exc)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc_info):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return False


def _lock_for(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _send(update,
        "👋 Hey! I'm your personal movie bot.\n\n"
        "1. /setletterboxd <username> — link your Letterboxd\n"
        "2. /rec <vibe> — get recommendations\n"
        "3. Or just message me a vibe directly!\n\n"
        "You can name a film you loved and I'll find things genuinely like it:\n"
        "  /rec something like Interstellar\n"
        "  /rec dark and mind-bending\n\n"
        "👍/👎 the picks I send and I'll learn from it — that works even "
        "without a Letterboxd account.\n\n"
        "⏱️ Heads up: linking Letterboxd takes 30–60 seconds the first time "
        "while I read your ratings and look every film up. I'll tell you when "
        "I'm done — it's not frozen. After that everything is fast.\n\n"
        "/status tells you where I'm at."
    )


async def set_letterboxd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await _send(update, "Please provide your username!\nExample: /setletterboxd dave")
        return

    username    = context.args[0].strip().lstrip("@")
    telegram_id = str(update.effective_user.id)

    # Say something before the slow part, not after it.
    await _send(update,
        f"🔍 Looking up letterboxd.com/{username} …\n"
        f"This takes a few seconds — hang tight.")

    async with _Typing(update.message.chat):
        profile = await asyncio.to_thread(load_profile, username)

    if not profile:
        await _send(update,
            f"I couldn't find letterboxd.com/{username} — check the spelling "
            f"and make sure the profile is public.")
        return

    user_profiles[telegram_id] = username
    await asyncio.to_thread(_save_users)

    rated = profile["total_rated"]
    await _send(update,
        f"✅ Linked! Found {profile['total_films']} films "
        f"({rated} rated) on letterboxd.com/{username}\n\n"
        f"⏳ Now looking up each of your rated films so I can learn your taste. "
        f"This runs in the background and takes about 30–60 seconds — "
        f"I'll message you the moment it's ready.\n\n"
        f"You can ask me for something right now if you like; it'll just be "
        f"a bit slower until the warm-up finishes.")

    context.application.create_task(
        _warm_and_notify(username, update.effective_chat.id, context)
    )


async def _warm_and_notify(username: str, chat_id: int,
                           context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resolve every film in a profile, then tell the user it's done."""
    if username in _warming:
        return
    _warming.add(username)
    started = time.time()
    try:
        summary = await asyncio.to_thread(warm_profile, username)
    except Exception as exc:
        log.warning("warm-up failed for %s: %s", username, exc, exc_info=True)
        await context.bot.send_message(
            chat_id,
            "⚠️ I had trouble reading your full history, but I can still "
            "recommend things — just ask.")
        return
    finally:
        _warming.discard(username)

    if not summary.get("ok"):
        return

    elapsed = time.time() - started
    genres = ", ".join(summary.get("top_genres") or []) or "a bit of everything"
    log.info("warmed %s in %.1fs (%d/%d resolved)",
             username, elapsed, summary["resolved"], summary["total"])

    await context.bot.send_message(
        chat_id,
        f"✅ All set — I've mapped {summary['resolved']} of your "
        f"{summary['total']} rated films ({elapsed:.0f}s).\n"
        f"You watch a lot of: {genres}.\n\n"
        f"Ask away — try /rec something like Interstellar")


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = user_profiles.get(str(update.effective_user.id))
    await _send(update,
        f"You're linked to letterboxd.com/{username}" if username
        else "You haven't linked a Letterboxd account yet — /setletterboxd <username>")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Let the user see the bot is working rather than guess."""
    username = user_profiles.get(str(update.effective_user.id))
    lines = [f"🎬 Library: {cache.count(current_only=True)} films indexed"]

    signal = load_default()
    if signal is None:
        lines.append("⚠️ Scoring: no model loaded — run python -m src.train")
    else:
        names = {"metadata": "metadata", "semantic": "meaning",
                 "collab": "what other viewers liked"}
        active = [names.get(s.name, s.name) for s in getattr(signal, "signals", [])]
        lines.append("🧠 Matching on: " + ", ".join(active))

    votes = await asyncio.to_thread(feedback.summary, str(update.effective_user.id))
    lines.append(f"👍 Your ratings: {votes}")

    if not username:
        lines.append("🔗 Letterboxd: not linked (/setletterboxd <username>)")
    elif username in _warming:
        lines.append(f"⏳ Letterboxd: reading {username}'s history right now — "
                     f"recommendations work, they're just slower until it finishes.")
    else:
        age = await asyncio.to_thread(cache.profile_age, username)
        if age is None:
            lines.append(f"🔗 Letterboxd: {username} linked, not read yet "
                         f"(your next request will read it)")
        else:
            lines.append(f"✅ Letterboxd: {username}, read "
                         f"{int(age // 60)} min ago — everything's warm")
    await _send(update, "\n".join(lines))


async def refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Re-read a linked profile after the user has logged new films."""
    telegram_id = str(update.effective_user.id)
    username = user_profiles.get(telegram_id)
    if not username:
        await _send(update, "Link an account first — /setletterboxd <username>")
        return

    await _send(update, f"🔄 Re-reading letterboxd.com/{username} — one moment…")
    async with _Typing(update.message.chat):
        profile = await asyncio.to_thread(load_profile, username, True, True)

    if not profile:
        await _send(update, "Couldn't re-read that profile just now — try again shortly.")
        return

    await _send(update, f"✅ Refreshed: {profile['total_films']} films "
                        f"({profile['total_rated']} rated).")
    context.application.create_task(
        _warm_and_notify(username, update.effective_chat.id, context))


async def _respond(update: Update, message: str) -> None:
    telegram_id = str(update.effective_user.id)
    username    = user_profiles.get(telegram_id)

    now = time.time()
    if now - _last_request.get(telegram_id, 0) < MIN_SECONDS_BETWEEN:
        await _send(update, "One at a time! Give me a couple of seconds 🙂")
        return
    _last_request[telegram_id] = now

    lock = _lock_for(telegram_id)
    if lock.locked():
        await _send(update, "Still working on your last one — hang on ⏳")
        return

    # If we're about to do the slow cold-start work, say so first.
    if username:
        if username in _warming:
            await _send(update,
                "⏳ I'm still reading your Letterboxd history — this one may take "
                "a little longer than usual. Working on it…")
        else:
            age = await asyncio.to_thread(cache.profile_age, username)
            if age is None:
                await _send(update,
                    "⏳ First request since you linked — I need to read your "
                    "Letterboxd history, so this one takes 30–60 seconds. "
                    "Everything after it is fast.")

    async with lock, _Typing(update.message.chat):
        try:
            # Off the event loop: this call does blocking HTTP and sklearn work.
            result = await asyncio.to_thread(
                recommend_message, message, username, telegram_id)
        except Exception as exc:
            log.exception("recommendation failed: %s", exc)
            result = None

    if result is None:
        await _send(update, "Something went wrong on my end — try again in a moment!")
        return

    await _send(update, result.text, _vote_keyboard(result.picks))


async def rec(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await _send(update, "Tell me a vibe!\nExample: /rec something dark and mind-bending")
        return
    await _respond(update, " ".join(context.args))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _respond(update, update.message.text)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("unhandled error", exc_info=context.error)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config.require("BOT_TOKEN", "TMDB_TOKEN", "GROQ_API_KEY")

    app = ApplicationBuilder().token(config.BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("setletterboxd", set_letterboxd))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("refresh", refresh))
    app.add_handler(CommandHandler("rec", rec))
    app.add_handler(CallbackQueryHandler(on_vote, pattern=f"^{VOTE_PREFIX}:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(on_error)

    # Load the scoring signals before serving anyone. They take a second
    # or two to read off disk, and paying that inside somebody's first
    # request is exactly the kind of unexplained wait this bot used to
    # be full of.
    signal = load_default()
    if signal is None:
        log.warning("no scoring model found - run: python -m src.train "
                    "&& python -m src.embed && python -m src.collab")
    else:
        log.info("scoring signals ready: %s",
                 ", ".join(s.name for s in getattr(signal, "signals", [])))

    log.info("bot starting (%d linked users, %d films indexed)",
             len(user_profiles), cache.count(current_only=True))
    app.run_polling()


if __name__ == "__main__":
    main()
