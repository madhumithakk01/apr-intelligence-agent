"""FastAPI entrypoint -- SPEC.md section 15.

The portfolio-scale pipeline (SPEC.md section 5) is an async job:
submit, poll, resume. This module wires the DB, mounts the three API
routers, and does nothing else -- the single-record `/analyze` path and
its legacy scoring/report services were removed once the async pipeline
fully replaced them (SPEC.md sections 4.2, 4.8, 5).
"""

from dotenv import load_dotenv
from fastapi import FastAPI

import app.database.models  # noqa: F401 -- register models for create_all
from app.api.batch import router as batch_router
from app.api.retention import router as retention_router
from app.api.shadow import router as shadow_router
from app.database.db import Base, engine, migrate_schema

load_dotenv()

Base.metadata.create_all(bind=engine)
migrate_schema()

app = FastAPI(title="APR Intelligence Agent", version="1.0.0")

app.include_router(batch_router)
app.include_router(shadow_router)
app.include_router(retention_router)


@app.get("/")
def root() -> dict:
    return {
        "service": "APR Intelligence Agent",
        "docs": "/docs",
        "submit_a_run": "POST /api/runs",
        "poll_a_run": "GET /api/runs/{run_id}",
        "shadow_status": "GET /api/shadow",
    }
