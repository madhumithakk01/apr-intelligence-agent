"""HTTP layer for the async batch job -- CLAUDE.md sections 11, 14.

Thin: it validates input, calls app.orchestration.batch_runner, and
maps its two error types onto status codes. All the run mechanics live
in the runner. Mounted by app.main via ``include_router``.

  POST /api/runs          submit rows (JSON)          -> 202, run view
  POST /api/runs/upload    submit an .xlsx workbook   -> 202, run view
  GET  /api/runs           list runs
  GET  /api/runs/{id}      poll one run
  POST /api/runs/{id}/resume   feed a gate decision   -> run view
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.orchestration.batch_runner import (
    BatchRunner,
    RunNotFoundError,
    RunNotResumableError,
    get_runner,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "knowledge_db" / "batch_uploads"

router = APIRouter(prefix="/api/runs", tags=["batch"])


class RunSubmission(BaseModel):
    applications: List[Dict[str, Any]] = Field(default_factory=list)
    data_sensitivity: Literal["real", "synthetic"] = "real"
    """CLAUDE.md section 11: defaults to "real" so an omitted flag fails
    closed -- the Groq-only routing applies unless a caller deliberately
    declares synthetic fixtures."""


class ResumeRequest(BaseModel):
    decision: Any
    """Passed to the gate verbatim (app.orchestration.gates records it
    as-is). A rubric sign-off is "approved"/"rejected" or a dict; the
    queue-driven gates 2-5 accept any shape."""


def _runner_dep() -> BatchRunner:
    return get_runner()


def _view_or_404(runner: BatchRunner, run_id: str) -> Dict[str, Any]:
    view = runner.get(run_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return view


@router.post("", status_code=202)
def submit_run(body: RunSubmission, runner: BatchRunner = Depends(_runner_dep)) -> Dict[str, Any]:
    run_id = runner.submit(body.applications, data_sensitivity=body.data_sensitivity)
    return _view_or_404(runner, run_id)


@router.post("/upload", status_code=202)
def submit_run_from_file(
    file: UploadFile = File(...),
    data_sensitivity: Literal["real", "synthetic"] = Form("real"),
    runner: BatchRunner = Depends(_runner_dep),
) -> Dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="expected an .xlsx/.xls workbook")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{uuid4().hex}_{Path(filename).name}"
    destination.write_bytes(file.file.read())
    run_id = runner.submit([], data_sensitivity=data_sensitivity, dataset_path=str(destination))
    return _view_or_404(runner, run_id)


@router.get("")
def list_runs(runner: BatchRunner = Depends(_runner_dep)) -> Dict[str, Any]:
    return {"runs": runner.list_runs()}


@router.get("/{run_id}")
def get_run(run_id: str, runner: BatchRunner = Depends(_runner_dep)) -> Dict[str, Any]:
    return _view_or_404(runner, run_id)


@router.post("/{run_id}/resume")
def resume_run(
    run_id: str, body: ResumeRequest, runner: BatchRunner = Depends(_runner_dep)
) -> Dict[str, Any]:
    try:
        runner.resume(run_id, body.decision)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    except RunNotResumableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _view_or_404(runner, run_id)
