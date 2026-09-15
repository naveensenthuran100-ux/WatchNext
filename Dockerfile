# Containerised alternative to the systemd service (see DEPLOY.md).
# A pinned Python base so the host's Python version is irrelevant. Builds
# on ARM (Oracle Ampere A1) and x86 alike.
FROM python:3.13-slim

WORKDIR /app

# Dependencies first, so a code change does not re-download every wheel.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code only. The model files and SQLite live in a mounted volume at
# /app/data (see DEPLOY.md) so votes and cached profiles survive a
# redeploy instead of being baked into the image and lost.
COPY src ./src

# .env is passed at run time with --env-file, never copied into the image.
CMD ["python", "-m", "src.bot"]
