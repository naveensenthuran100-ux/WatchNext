
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from src.recommender import get_vibe_recommendations

load_dotenv()


user_profiles = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hey! I'm your personal movie bot.\n\n"
        "Here's how to use me:\n"
        "1. /setletterboxd <username> — link your Letterboxd\n"
        "2. /rec <vibe> — get personalised recommendations\n"
        "3. Or just message me a vibe directly!\n\n"
        "Example: /rec something dark and mind-bending"
    )

async def set_letterboxd(update: Update, context: ContextTypes.DEFAULT_TYPE):
   
    if not context.args:
        await update.message.reply_text(
            "Please provide your username!\n"
            "Example: /setletterboxd dave"
        )
        return

    letterboxd_username = context.args[0]
    telegram_id         = str(update.effective_user.id)

    user_profiles[telegram_id] = letterboxd_username

    await update.message.reply_text(
        f"✅ Linked! I'll now personalise recommendations for "
        f"letterboxd.com/{letterboxd_username}\n\n"
        f"Try: /rec something dark and mind-bending"
    )

async def rec(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Tell me a vibe!\n"
            "Example: /rec something dark and mind-bending"
        )
        return

    vibe        = " ".join(context.args)  
    telegram_id = str(update.effective_user.id)
    username    = user_profiles.get(telegram_id)  

    await update.message.reply_text("🎬 Finding your films...")

    try:
        response = get_vibe_recommendations(vibe, username=username)
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        print(f"[bot] Error: {e}")
        await update.message.reply_text(
            "Something went wrong — try again in a moment!"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vibe        = update.message.text
    telegram_id = str(update.effective_user.id)
    username    = user_profiles.get(telegram_id)

    await update.message.reply_text("🎬 Finding your films...")

    try:
        response = get_vibe_recommendations(vibe, username=username)
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        print(f"[bot] Error: {e}")
        await update.message.reply_text(
            "Something went wrong — try again in a moment!"
        )

def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN not found in .env")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start",          start))
    app.add_handler(CommandHandler("setletterboxd",  set_letterboxd))
    app.add_handler(CommandHandler("rec",            rec))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("[bot] Starting...")
    app.run_polling()

if __name__ == "__main__":
    main()