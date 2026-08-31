# APR Intelligence Agent

Application Portfolio Rationalization (APR) tool for a competitive
pre-sales pitch. Phase 1 (this system) analyses a single Excel export of
~100 applications the client has shared under bid confidentiality, where
some fields are deliberately withheld.

It is a portfolio-scale batch pipeline: rationalisation is inherently
cross-portfolio (redundancy across departments, cost outliers relative
to peers), so the whole file is processed together, not row by row.

`CLAUDE.md` is the architecture specification and the source of truth
for every rule and parameter below.

## Pipeline

A single LangGraph run, submitted as an async job (a ~100-row run is
rate-limited well past any request timeout):

1. **Ingestion** — deterministic Excel load, duplicate-ID surfacing,
   safe numeric parsing.
2. **Disclosure & provenance classification** — one structured LLM call
   per field: `Answered` / `Withheld-Confidential` / `Deferred-until-award`
   / `Genuinely-unknown` / `Suspicious-placeholder`. Gates every
   downstream step; doubles as the Phase 2 discovery agenda.
3. **Rubric calibration** — one call per field per engagement, frozen by
   a human sign-off gate before any row is scored.
4. **Qualitative row scoring** — single call, escalating to a 3-sample
   ensemble on low confidence or rubric mismatch.
5. **Deterministic scoring kernel** — one TIM-E / COTS-fit engine
   (Gartner TIME), with non-compensatory floors.
6. **Redundancy** — generous capability blocking, a multi-axis profile
   per app, a 3-sample adjudication ensemble into a five-way typology,
   and an ordered non-compensatory recommendation policy.
7. **Cost-outlier detection** — deterministic IQR flag, single LLM call
   to judge explainability.
8. **Market intelligence** — the one genuine agent: a LangGraph search
   loop, fanned out per redundancy-surviving segment.
9. **Product extraction & grounding** — one structured call plus a
   deterministic claim-by-claim substring check against retrieved text.
10. **Narrative generation** — one call with a fixed one-retry budget
    and a deterministic structured-bullet fallback.
11. **Report rendering** — one deterministic renderer to a structured
    dict plus Markdown.

Five human-in-the-loop gates (`interrupt()`), LangSmith tracing on every
LLM call, a shadow-mode delivery gate (nothing is client-deliverable
until a signed-off shadow run), and a bid-outcome data purge.

## Running

```bash
uvicorn app.main:app --reload
# POST /api/runs        submit a run (JSON rows or /upload an .xlsx)
# GET  /api/runs/{id}   poll; resume gates via POST /api/runs/{id}/resume
# GET  /api/shadow      the shadow-mode delivery gate
# POST /api/retention/purge   the bid-outcome data purge
```

Load the client workbook into the local DB for inspection with
`python import_dataset.py`. Docker, CI, Render, and the Postgres
migration path are in `docs/deployment.md`.

## Tests

```bash
pytest tests/ -v
pytest tests/golden_subset -v   # the required CI regression gate
ruff check .
```

## Stack

Python 3.12, FastAPI, LangGraph, SQLAlchemy + SQLite, pandas / openpyxl,
Groq (primary LLM) with a synthetic-only Gemini fallback, Tavily search,
LangSmith tracing.
