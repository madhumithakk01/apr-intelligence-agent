# Deployment

CLAUDE.md section 15. The service is a single FastAPI app
(`app.main:app`) exposing the async batch job API (`/api/runs`) plus the
legacy single-record routes.

## Run locally with Docker

```bash
docker build -t apr-intelligence-agent .
docker run --rm -p 8000:8000 --env-file .env apr-intelligence-agent
# -> http://localhost:8000/api/runs
```

The image is `python:3.12-slim`, single stage, uvicorn entrypoint. It
copies only `app/` and `templates/`; `.dockerignore` keeps tests, local
state, secrets, and dev scripts out of the build context.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Primary LLM provider. Without it the pipeline cannot score. |
| `TAVILY_API_KEY` | for market research | Search tool for the market intelligence agent. |
| `GEMINI_API_KEY` | no | Rate-limit-only fallback, **synthetic data only** (CLAUDE.md section 11). |
| `LANGSMITH_API_KEY` | no | Call tracing. Absent -> `app.llm.tracing` is a silent no-op. |
| `LANGSMITH_PROJECT` | no | LangSmith project name (default `apr-intelligence-agent`). |
| `PORT` | injected by host | uvicorn bind port; defaults to 8000. |

Never bake `.env` into the image or commit it.

## Render (free tier)

`render.yaml` is a Blueprint: one Docker web service, `plan: free`,
health check on `/api/runs`, `autoDeploy: false`. Create the service
from the blueprint, then set every `sync: false` key in the dashboard.

## Persistence — the free-tier limitation

Two SQLite files are written at runtime and **both live on the
container's ephemeral disk**, which is wiped on every restart and
redeploy on the free tier:

- `knowledge_db/apr.db` — the application DB (`app/database/db.py`).
- `knowledge_db/orchestration_checkpoints.db` — the LangGraph checkpoint
  store (`app/orchestration/checkpointer.py`), plus a per-run market
  subgraph checkpoint DB.

Consequence: an in-flight run that is suspended at a human gate does not
survive a restart. `BatchRunner.get` reconstructs a coarse status from
the checkpoint when the file is still there, but a wiped disk loses it.

### Postgres migration path

Both stores move to Postgres at the same time, at two swap points:

1. `app/database/db.py` — replace the `sqlite:///` URL with
   `os.environ["DATABASE_URL"]` (Render Postgres provides it) and drop
   the `check_same_thread` connect arg.
2. `app/orchestration/checkpointer.py` — swap `SqliteSaver` for
   `langgraph.checkpoint.postgres.PostgresSaver` over the same
   `DATABASE_URL`. `build_in_memory_checkpointer` and
   `purge_checkpoint_store` stay as they are.

No pipeline code changes: every stage already treats the checkpointer as
opaque.

## CI

- `.github/workflows/golden-subset.yml` — the required check: the
  deterministic golden-subset regression suite, no secrets.
- `.github/workflows/ci.yml` — lint (`ruff check .`), the full test
  suite (`pytest tests/`), and a `docker build` of this image.

## Data deletion (CLAUDE.md section 2)

Cached and stored client data has a deletion trigger tied to the bid
outcome, not a TTL. On that trigger, clear all of:

- `purge_checkpoint_store()` for each checkpoint DB;
- the application DB rows / file;
- the LangSmith project (out of band, in LangSmith);
- any uploaded workbooks under `knowledge_db/batch_uploads/` and
  generated files under `reports/`.
