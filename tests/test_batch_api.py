"""HTTP layer for the async batch job -- SPEC.md sections 11, 14.

Mounts only app.api.batch.router on a bare FastAPI app (never
app.main, which pulls the legacy service chain and touches the real DB)
and overrides the runner dependency with a synchronous one over an
in-memory checkpointer.
"""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import batch
from app.orchestration.batch_runner import BatchRunner
from app.orchestration.checkpointer import build_in_memory_checkpointer


@pytest.fixture
def client():
    runner = BatchRunner(checkpointer=build_in_memory_checkpointer(), submit_fn=lambda fn: fn())
    app = FastAPI()
    app.include_router(batch.router)
    app.dependency_overrides[batch._runner_dep] = lambda: runner
    with TestClient(app) as test_client:
        test_client.runner = runner
        yield test_client


def _submit(client, **body):
    body.setdefault("applications", [])
    body.setdefault("data_sensitivity", "synthetic")
    return client.post("/api/runs", json=body)


# --- submit / poll / resume ------------------------------------------


def test_submit_returns_202_and_a_run_awaiting_review(client):
    response = _submit(client)
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"]
    assert body["status"] == "awaiting_review"
    assert body["pending_review"]["gate"] == "gate_1_rubric_signoff"


def test_poll_then_resume_drives_the_run_to_completion(client):
    run_id = _submit(client).json()["run_id"]

    polled = client.get(f"/api/runs/{run_id}")
    assert polled.status_code == 200
    assert polled.json()["status"] == "awaiting_review"

    resumed = client.post(f"/api/runs/{run_id}/resume", json={"decision": "approved"})
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "complete"
    assert body["gates_completed"] == ["gate_1_rubric_signoff"]
    assert body["report"]["markdown"].startswith("# APR Portfolio Rationalization Report")


def test_list_runs_reports_every_submitted_run(client):
    ids = {_submit(client).json()["run_id"] for _ in range(2)}
    listed = client.get("/api/runs").json()["runs"]
    assert {r["run_id"] for r in listed} == ids


# --- error mapping -------------------------------------------------


def test_polling_an_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_resuming_an_unknown_run_is_404(client):
    assert client.post("/api/runs/nope/resume", json={"decision": "approved"}).status_code == 404


def test_resuming_a_completed_run_is_409(client):
    run_id = _submit(client).json()["run_id"]
    client.post(f"/api/runs/{run_id}/resume", json={"decision": "approved"})

    again = client.post(f"/api/runs/{run_id}/resume", json={"decision": "approved"})
    assert again.status_code == 409
    assert "not awaiting review" in again.json()["detail"]


# --- data-sensitivity routing flag (SPEC.md section 11) -----------


def test_data_sensitivity_defaults_to_real_when_omitted(client):
    response = client.post("/api/runs", json={"applications": []})
    assert response.status_code == 202
    assert response.json()["data_sensitivity"] == "real"


def test_an_unrecognized_data_sensitivity_is_rejected_422(client):
    response = client.post("/api/runs", json={"applications": [], "data_sensitivity": "leaky"})
    assert response.status_code == 422


# --- upload ------------------------------------------------------


def test_upload_rejects_a_non_workbook_file(client):
    response = client.post(
        "/api/runs/upload",
        files={"file": ("notes.txt", io.BytesIO(b"not a workbook"), "text/plain")},
    )
    assert response.status_code == 400
