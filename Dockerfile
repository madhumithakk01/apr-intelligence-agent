# APR Intelligence Agent -- CLAUDE.md section 15.
# Single stage, python:3.12-slim, uvicorn entrypoint. The pipeline is
# I/O-bound (rate-limited LLM and search calls), so one worker suits the
# Render free tier; raise WEB_CONCURRENCY when it moves off it.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: a code-only change must not reinstall everything.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY templates ./templates

# Runtime-writable state the app creates on import (SQLite app DB and the
# LangGraph checkpoint store live here). On Render's free tier this
# filesystem is ephemeral across restarts and redeploys -- a documented
# limitation, with Postgres as the stated migration path for both
# (docs/deployment.md).
RUN mkdir -p knowledge_db reports

EXPOSE 8000

# Render and most PaaS inject $PORT; default to 8000 for a bare
# `docker run -p 8000:8000`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
