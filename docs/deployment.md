# Deployment

CLAUDE.md section 15. The service is a single FastAPI app
(`app.main:app`) exposing the async batch job API: `/api/runs` (submit /
poll / resume), `/api/shadow` (the shadow-mode delivery gate), and
`/api/retention` (the bid-outcome purge).

## Run locally with Docker

```bash
docker build -t apr-intelligence-agent .
docker run --rm -p 8000:8000 --env-file .env apr-intelligence-agent
# -> http://localhost:8000/api/runs
```

The image is `python:3.12-slim`, single stage, uvicorn entrypoint. It
copies only `app/`; `.dockerignore` keeps tests, local state, secrets,
and dev scripts out of the build context.

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

## Shadow mode (CLAUDE.md section 2)

Nothing the pipeline produces is client-deliverable until a full run has
executed in **shadow mode** and an internal reviewer has signed it off.

- Every run defaults to `run_mode: "shadow"`. `POST /api/runs` and
  `/api/runs/upload` accept `"shadow"` or `"live"`.
- A shadow run's report is stamped `delivery.client_deliverable: false`
  and the Markdown carries a `SHADOW RUN -- INTERNAL REVIEW ONLY`
  banner. A `live` run is stamped non-deliverable too until the
  engagement is unlocked.
- `GET /api/shadow` shows engagement status; `POST /api/shadow/signoff`
  (`{run_id, reviewer, decision}`) records a reviewer's verdict. An
  approved sign-off of a completed shadow run unlocks
  `client_deliverable: true` for subsequent `live` runs.
- Engagement state is a JSON ledger at `SHADOW_LEDGER_PATH` (default
  `knowledge_db/shadow_ledger.json`) -- ephemeral on the free tier, same
  Postgres migration point as everything else.

## CI

- `.github/workflows/golden-subset.yml` — the required check: the
  deterministic golden-subset regression suite, no secrets.
- `.github/workflows/ci.yml` — lint (`ruff check .`), the full test
  suite (`pytest tests/`), and a `docker build` of this image.

## Data deletion (CLAUDE.md section 2)

Cached and stored client data has one deletion trigger -- the bid
concluded -- not a TTL. `app.retention.purge.purge_all_client_data` is
the single entry point: it clears every store together and writes an
audit line to `knowledge_db/purge_audit.jsonl`.

```bash
python -m app.retention.purge --reason "bid awarded to vendor X" --dry-run   # preview
python -m app.retention.purge --reason "bid awarded to vendor X" --confirm   # do it
```

or `POST /api/retention/purge` with `{"reason": "...", "confirm": true}`
(`"dry_run": true` to preview); `GET /api/retention/purge/history` shows
recent audit records.

It clears: both LangGraph checkpoint stores (with their `-wal`/`-shm`
sidecars), the `applications` / `market_products` / `analysis_runs` DB
rows, the shadow-mode ledger, uploaded workbooks under
`knowledge_db/batch_uploads/`, and generated files under `reports/`.

**LangSmith runs are not deleted by this** -- they live in a
third-party project. Delete the `LANGSMITH_PROJECT` project in LangSmith
manually; the purge report flags this as an outstanding step.
