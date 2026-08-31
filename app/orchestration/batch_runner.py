"""Async batch job runner -- SPEC.md sections 11, 14, 15.

A ~100-row portfolio run makes hundreds of rate-limited LLM calls and
can stop at up to five human gates, so it cannot be a blocking request
(SPEC.md section 11). This is the "submit -> run id -> poll -> resume"
machinery around app.orchestration.graph:

  submit  registers a run, kicks execution onto a worker, returns now
  poll    returns a status snapshot
          (queued | running | awaiting_review | complete | failed)
  resume  feeds one gate's reviewer decision back in and re-kicks it

Run status lives in an in-process registry; the resumable graph state
lives in the LangGraph checkpointer (SQLite now, Postgres the stated
production path -- the same migration point as the app DB, SPEC.md
section 15). After a process restart the registry is empty but the
checkpoint file survives, so ``get`` falls back to reconstructing a
coarse status from ``graph.get_state`` -- a client that kept its run id
can still poll and resume.

The runner is deterministic to test: pass ``submit_fn=lambda fn: fn()``
for synchronous execution and an in-memory checkpointer.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from langgraph.types import Command

from app.orchestration.checkpointer import build_sqlite_checkpointer
from app.orchestration.graph import build_graph, initial_state, run_config
from app.orchestration.shadow import normalize_mode

QUEUED = "queued"
RUNNING = "running"
AWAITING_REVIEW = "awaiting_review"
COMPLETE = "complete"
FAILED = "failed"

_GATE_NODES = {
    "gate_rubric_signoff",
    "gate_qualitative_disagreement",
    "gate_redundancy_verdict",
    "gate_cost_outlier",
    "gate_narrative_grounding",
}


class RunNotFoundError(KeyError):
    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.run_id = run_id


class RunNotResumableError(RuntimeError):
    def __init__(self, run_id: str, status: str):
        super().__init__(f"run {run_id} is {status}, not awaiting review")
        self.run_id = run_id
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _RunRecord:
    run_id: str
    data_sensitivity: str
    application_count: int
    submitted_at: str
    updated_at: str
    run_mode: str = "shadow"
    status: str = QUEUED
    pending_review: Optional[Dict[str, Any]] = None
    gates_completed: List[str] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def view(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "run_mode": self.run_mode,
            "data_sensitivity": self.data_sensitivity,
            "application_count": self.application_count,
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
            "gates_completed": list(self.gates_completed),
            "pending_review": self.pending_review,
            "delivery": (self.report or {}).get("delivery"),
            "report": self.report,
            "error": self.error,
            "recovered": False,
        }


_executor: Optional[ThreadPoolExecutor] = None


def _default_submit(fn: Callable[[], None]) -> None:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="apr-batch")
    _executor.submit(fn)


class BatchRunner:
    def __init__(
        self,
        *,
        checkpointer: Any = None,
        graph: Any = None,
        submit_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ):
        self._checkpointer = checkpointer if checkpointer is not None else build_sqlite_checkpointer()
        self._graph = graph if graph is not None else build_graph(self._checkpointer)
        self._submit_fn = submit_fn or _default_submit
        self._registry: Dict[str, _RunRecord] = {}
        self._lock = threading.Lock()

    # --- public API -----------------------------------------------------

    def submit(
        self,
        applications: Optional[List[Dict[str, Any]]],
        *,
        data_sensitivity: str = "real",
        dataset_path: Optional[str] = None,
        run_mode: str = "shadow",
    ) -> str:
        applications = list(applications or [])
        run_id = uuid4().hex
        run_mode = normalize_mode(run_mode)
        record = _RunRecord(
            run_id=run_id,
            data_sensitivity=data_sensitivity,
            run_mode=run_mode,
            application_count=len(applications),
            submitted_at=_now(),
            updated_at=_now(),
        )
        with self._lock:
            self._registry[run_id] = record

        state = initial_state(run_id, applications, data_sensitivity, run_mode)
        if dataset_path:
            state = {**state, "dataset_path": dataset_path}
        self._schedule(run_id, state)
        return run_id

    def resume(self, run_id: str, decision: Any) -> None:
        with self._lock:
            record = self._registry.get(run_id)
            if record is None:
                raise RunNotFoundError(run_id)
            if record.status != AWAITING_REVIEW:
                raise RunNotResumableError(run_id, record.status)
            if record.pending_review and record.pending_review.get("gate"):
                record.gates_completed.append(record.pending_review["gate"])
            record.pending_review = None
        self._schedule(run_id, Command(resume=decision))

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._registry.get(run_id)
        if record is not None:
            return record.view()
        return self._recover_from_checkpoint(run_id)

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [record.view() for record in self._registry.values()]

    # --- execution ----------------------------------------------------

    def _schedule(self, run_id: str, invoke_input: Any) -> None:
        with self._lock:
            record = self._registry[run_id]
            record.status = RUNNING
            record.updated_at = _now()
        self._submit_fn(lambda: self._drive(run_id, invoke_input))

    def _drive(self, run_id: str, invoke_input: Any) -> None:
        try:
            result = self._graph.invoke(invoke_input, run_config(run_id))
        except Exception as exc:  # noqa: BLE001 -- a failed run is a status, never a crash
            self._finish(run_id, status=FAILED, error=f"{type(exc).__name__}: {exc}")
            return

        interrupts = list(result.get("__interrupt__") or [])
        if interrupts:
            self._finish(run_id, status=AWAITING_REVIEW, pending_review=dict(interrupts[0].value))
        else:
            self._finish(run_id, status=COMPLETE, report=dict(result.get("report") or {}))

    def _finish(
        self,
        run_id: str,
        *,
        status: str,
        pending_review: Optional[Dict[str, Any]] = None,
        report: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._registry[run_id]
            record.status = status
            record.updated_at = _now()
            record.pending_review = pending_review
            if report is not None:
                record.report = report
            if error is not None:
                record.error = error

    # --- restart recovery -------------------------------------------

    def _recover_from_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Best effort: after a restart the registry is gone but the
        checkpoint is not. Reconstruct a coarse view so a client that
        kept its run id is not stranded. The full pending-review payload
        and any failure text are not recoverable this way."""
        try:
            snapshot = self._graph.get_state(run_config(run_id))
        except Exception:  # noqa: BLE001
            return None
        if snapshot is None or (not snapshot.values and not snapshot.next):
            return None

        next_nodes = set(snapshot.next or ())
        if not next_nodes:
            status = COMPLETE
        elif next_nodes & _GATE_NODES:
            status = AWAITING_REVIEW
        else:
            status = RUNNING

        report = snapshot.values.get("report") if status == COMPLETE else None
        return {
            "run_id": run_id,
            "status": status,
            "run_mode": normalize_mode(snapshot.values.get("run_mode")),
            "data_sensitivity": snapshot.values.get("data_sensitivity"),
            "application_count": len(snapshot.values.get("applications") or []),
            "submitted_at": None,
            "updated_at": None,
            "gates_completed": sorted((snapshot.values.get("gate_decisions") or {}).keys()),
            "pending_review": None,
            "delivery": (report or {}).get("delivery"),
            "report": report,
            "error": None,
            "recovered": True,
        }


_runner: Optional[BatchRunner] = None


def get_runner() -> BatchRunner:
    """Process-wide singleton used by the HTTP layer (app.api.batch).
    Tests build their own BatchRunner instead."""
    global _runner
    if _runner is None:
        _runner = BatchRunner()
    return _runner
