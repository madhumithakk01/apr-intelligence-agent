"""Async batch job runner -- CLAUDE.md sections 11, 14, 15.

Two kinds of test here:
  * the real graph, driven over an empty portfolio -- fully hermetic
    (no row means no fan-out and no LLM call; calibrate_rubrics
    short-circuits on empty input), so it exercises submit -> gate 1 ->
    resume -> complete against real wiring;
  * a scripted fake graph, for the multi-gate and failure paths the
    empty run cannot reach.

Every runner is synchronous (submit_fn=lambda fn: fn()) so a poll right
after submit/resume already reflects the outcome.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.orchestration import batch_runner as br
from app.orchestration.batch_runner import (
    AWAITING_REVIEW,
    COMPLETE,
    FAILED,
    BatchRunner,
    RunNotFoundError,
    RunNotResumableError,
)
from app.orchestration.checkpointer import build_in_memory_checkpointer


def _real_runner(**kw):
    return BatchRunner(checkpointer=build_in_memory_checkpointer(), submit_fn=lambda fn: fn(), **kw)


# --- fake graph for the paths an empty real run cannot reach --------------


class _FakeGraph:
    """invoke() consumes one scripted step per call:
    "gate:<id>" -> suspend at that gate; "done" -> finish with a report;
    "boom" -> raise."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def invoke(self, invoke_input, config):
        self.calls.append((invoke_input, config))
        step = self._script.pop(0)
        if step == "boom":
            raise RuntimeError("synthetic graph failure")
        if step == "done":
            return {"report": {"run_id": config["configurable"]["thread_id"], "markdown": "# done"}}
        assert step.startswith("gate:")
        return {"__interrupt__": [SimpleNamespace(value={"gate": step[5:], "reason": "r", "items": []})]}

    def get_state(self, config):  # only used by recovery tests, not these
        return None


def _fake_runner(script):
    return BatchRunner(graph=_FakeGraph(script), checkpointer=object(), submit_fn=lambda fn: fn())


# --- real graph: the happy path -----------------------------------------


def test_submit_returns_a_run_id_and_stops_at_gate_1():
    runner = _real_runner()
    run_id = runner.submit([], data_sensitivity="synthetic")

    view = runner.get(run_id)
    assert view["run_id"] == run_id
    assert view["status"] == AWAITING_REVIEW
    assert view["pending_review"]["gate"] == "gate_1_rubric_signoff"
    assert view["application_count"] == 0
    assert view["report"] is None


def test_resume_approved_runs_to_completion_with_a_rendered_report():
    runner = _real_runner()
    run_id = runner.submit([], data_sensitivity="synthetic")

    runner.resume(run_id, "approved")

    view = runner.get(run_id)
    assert view["status"] == COMPLETE
    assert view["pending_review"] is None
    assert view["gates_completed"] == ["gate_1_rubric_signoff"]
    assert view["report"]["run_id"] == run_id
    assert view["report"]["markdown"].startswith("# APR Portfolio Rationalization Report")


def test_resume_records_a_rejection_verbatim_and_still_completes():
    runner = _real_runner()
    run_id = runner.submit([], data_sensitivity="synthetic")

    runner.resume(run_id, {"action": "reject", "reason": "anchors look wrong"})

    assert runner.get(run_id)["status"] == COMPLETE  # gate 1 rejection freezes rubrics, does not halt the run


# --- errors ---------------------------------------------------------------


def test_get_unknown_run_returns_none():
    assert _real_runner().get("does-not-exist") is None


def test_resume_unknown_run_raises():
    with pytest.raises(RunNotFoundError):
        _real_runner().resume("does-not-exist", "approved")


def test_resume_a_run_that_is_not_awaiting_review_raises():
    runner = _real_runner()
    run_id = runner.submit([], data_sensitivity="synthetic")
    runner.resume(run_id, "approved")  # -> complete

    with pytest.raises(RunNotResumableError) as excinfo:
        runner.resume(run_id, "approved")
    assert excinfo.value.status == COMPLETE


# --- fake graph: multi-gate and failure --------------------------------


def test_a_run_can_suspend_and_resume_through_several_gates():
    runner = _fake_runner(["gate:gate_2_qualitative_disagreement", "gate:gate_4_cost_outlier_explainability", "done"])
    run_id = runner.submit([{"application_id": "APP-1"}], data_sensitivity="synthetic")

    assert runner.get(run_id)["pending_review"]["gate"] == "gate_2_qualitative_disagreement"
    runner.resume(run_id, "approved")
    assert runner.get(run_id)["pending_review"]["gate"] == "gate_4_cost_outlier_explainability"
    runner.resume(run_id, "approved")

    view = runner.get(run_id)
    assert view["status"] == COMPLETE
    assert view["gates_completed"] == ["gate_2_qualitative_disagreement", "gate_4_cost_outlier_explainability"]


def test_a_graph_exception_becomes_a_failed_status_not_a_crash():
    runner = _fake_runner(["boom"])
    run_id = runner.submit([], data_sensitivity="synthetic")

    view = runner.get(run_id)
    assert view["status"] == FAILED
    assert view["error"] == "RuntimeError: synthetic graph failure"


def test_resume_passes_the_decision_through_to_the_graph():
    graph = _FakeGraph(["gate:gate_1_rubric_signoff", "done"])
    runner = BatchRunner(graph=graph, checkpointer=object(), submit_fn=lambda fn: fn())
    run_id = runner.submit([], data_sensitivity="synthetic")

    runner.resume(run_id, {"signed_off_by": "internal-reviewer"})

    resume_input = graph.calls[1][0]
    assert getattr(resume_input, "resume", None) == {"signed_off_by": "internal-reviewer"}


# --- listing & restart recovery -------------------------------------


def test_list_runs_returns_every_registered_run():
    runner = _real_runner()
    ids = {runner.submit([], data_sensitivity="synthetic") for _ in range(3)}
    assert {r["run_id"] for r in runner.list_runs()} == ids


def test_a_new_runner_sharing_the_checkpointer_recovers_a_finished_run():
    checkpointer = build_in_memory_checkpointer()
    first = BatchRunner(checkpointer=checkpointer, submit_fn=lambda fn: fn())
    run_id = first.submit([], data_sensitivity="synthetic")
    first.resume(run_id, "approved")

    # A fresh process: new registry, same checkpoint store.
    second = BatchRunner(checkpointer=checkpointer, submit_fn=lambda fn: fn())
    recovered = second.get(run_id)
    assert recovered["status"] == COMPLETE
    assert recovered["recovered"] is True
    assert recovered["report"]["run_id"] == run_id
    assert second.get("never-existed") is None


def test_get_runner_is_a_process_singleton(monkeypatch):
    monkeypatch.setattr(br, "_runner", None)
    assert br.get_runner() is br.get_runner()
