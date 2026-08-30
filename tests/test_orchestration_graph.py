"""Orchestration graph tests -- topology, resumability, gates, isolation.

This suite is about control flow, not what any one stage computes --
that is what each stage's own test module covers (disclosure
classification: tests/test_disclosure_classifier.py). Concretely:

  * every pipeline stage in CLAUDE.md section 5 is a node, in order
  * a run suspends at a gate and resumes from the checkpointer alone --
    including through a freshly compiled graph and a reopened SQLite
    file, which is what proves the state lives in the checkpoint rather
    than in the process
  * all five gates fire, each only on its own trigger
  * one fanned-out branch failing leaves its siblings' results intact
  * the node that wires each real stage to its module delegates
    correctly, without this file ever reaching a live provider

classify_disclosure, calibrate_rubrics, score_row, apply_scoring_kernel,
block_capabilities, build_profiles, and adjudicate_cluster have all made
real LLM calls (or deterministic module calls) since branches 6-10; only
detect_cost_outliers onward are still pass-through stubs. The autouse
fixture below replaces every LLM-calling hook with a deterministic fake
for this whole file, on purpose: it is what keeps a run of this suite
hermetic even on a machine that happens to have a real GROQ_API_KEY
exported in its shell. Tests that care about the real wiring override it
explicitly, which simply shadows the fixture for that one test.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from app.orchestration import gates, nodes
from app.orchestration.checkpointer import (
    build_in_memory_checkpointer,
    build_sqlite_checkpointer,
    purge_checkpoint_store,
)
from app.orchestration.graph import build_graph, initial_state, run_config
from app.orchestration.state import extend, merge_by_key
from app.rubric import calibration as rubric_calibration
from synthetic_fixtures import (
    SYNTHETIC_APPLICATIONS,
    SYNTHETIC_CLUSTERS,
    SYNTHETIC_PROFILES,
    SYNTHETIC_SEGMENTS,
)

_REAL_STAGE_DISCLOSURE = nodes._stage_disclosure
_REAL_CALIBRATE_RUBRICS = nodes.calibrate_rubrics
_REAL_STAGE_QUALITATIVE = nodes._stage_qualitative
_REAL_BLOCK_CAPABILITIES = nodes.block_capabilities
_REAL_BUILD_PROFILES = nodes.build_profiles
_REAL_STAGE_ADJUDICATION = nodes._stage_adjudication
_REAL_DETECT_COST_OUTLIERS = nodes.detect_cost_outliers
_REAL_EXPLAIN_COST_OUTLIERS = nodes.explain_cost_outliers
"""Captured at import time, before the autouse fixture below ever runs,
so a test that wants the real wiring back can restore it explicitly."""


def _fake_stage_disclosure(task):
    return {
        "results": {},
        "gated_application": dict(task.get("application") or {}),
        "phase2_agenda": [],
    }


def _fake_calibrate_rubrics(state):
    return {
        "rubrics": {"status": "proposed", "fields": {}},
        "stage_log": [nodes._stage("calibrate_rubrics")],
    }


def _fake_stage_qualitative(task):
    return {}


def _fake_block_capabilities(state):
    return {"clusters": list(state.get("clusters") or []), "stage_log": [nodes._stage("block_capabilities")]}


def _fake_build_profiles(state):
    return {"stage_log": [nodes._stage("build_profiles")]}


def _fake_stage_adjudication(task):
    return {}


def _fake_detect_cost_outliers(state):
    return {"cost_outliers": list(state.get("cost_outliers") or []), "stage_log": [nodes._stage("detect_cost_outliers")]}


def _fake_explain_cost_outliers(state):
    return {"stage_log": [nodes._stage("explain_cost_outliers")]}


@pytest.fixture(autouse=True)
def _no_real_llm_calls_in_this_file(monkeypatch):
    """This file tests orchestration control flow, not what
    classify_disclosure, calibrate_rubrics, score_row,
    block_capabilities, build_profiles, adjudicate_cluster, or
    explain_cost_outliers themselves compute (see
    tests/test_disclosure_classifier.py, tests/test_rubric_calibration.py,
    tests/test_qualitative_scoring.py, tests/test_blocking.py,
    tests/test_profile_builder.py, tests/test_redundancy_adjudicator.py,
    tests/test_recommendation_policy.py,
    tests/test_cost_outlier_detection.py, and
    tests/test_cost_outlier_explainability.py). Autouse so no test here
    can accidentally reach a real provider -- including on a machine that
    happens to have a real GROQ_API_KEY exported -- and so the rest of
    this file's tests can keep pre-setting state["clusters"]/
    state["profiles"] directly (_full_run_state below) without
    block_capabilities/build_profiles/detect_cost_outliers silently
    recomputing and overwriting them, or crashing on the placeholder
    SYNTHETIC_PROFILES shape when trying to deserialize it as a real
    ApplicationProfile. A test that cares about the real wiring restores
    the captured original above, which simply shadows this fixture's
    patch for that one test. apply_scoring_kernel,
    apply_recommendation_policy, and detect_cost_outliers need no
    provider fake for the calls they make (all three are fully
    deterministic); detect_cost_outliers still gets one here purely to
    avoid the profile-deserialization crash just described."""
    monkeypatch.setattr(nodes, "_stage_disclosure", _fake_stage_disclosure)
    monkeypatch.setattr(nodes, "calibrate_rubrics", _fake_calibrate_rubrics)
    monkeypatch.setattr(nodes, "_stage_qualitative", _fake_stage_qualitative)
    monkeypatch.setattr(nodes, "block_capabilities", _fake_block_capabilities)
    monkeypatch.setattr(nodes, "build_profiles", _fake_build_profiles)
    monkeypatch.setattr(nodes, "_stage_adjudication", _fake_stage_adjudication)
    monkeypatch.setattr(nodes, "detect_cost_outliers", _fake_detect_cost_outliers)
    monkeypatch.setattr(nodes, "explain_cost_outliers", _fake_explain_cost_outliers)

EXPECTED_NODES = {
    "ingest",
    "classify_disclosure",
    "calibrate_rubrics",
    "gate_rubric_signoff",
    "score_row",
    "gate_qualitative_disagreement",
    "apply_scoring_kernel",
    "block_capabilities",
    "build_profiles",
    "adjudicate_cluster",
    "gate_redundancy_verdict",
    "apply_recommendation_policy",
    "detect_cost_outliers",
    "explain_cost_outliers",
    "gate_cost_outlier",
    "research_segment",
    "extract_and_ground_products",
    "generate_narratives",
    "gate_narrative_grounding",
    "render_report",
}


def _full_run_state(**overrides):
    state = initial_state(
        run_id="run-synthetic",
        applications=SYNTHETIC_APPLICATIONS,
        data_sensitivity="synthetic",
    )
    state["clusters"] = list(SYNTHETIC_CLUSTERS)
    state["profiles"] = dict(SYNTHETIC_PROFILES)
    # Segments are produced by the redundancy stages (branch 10); with
    # those stubbed, the fan-out unit has to be supplied by the fixture.
    state["segments"] = list(SYNTHETIC_SEGMENTS)
    state.update(overrides)
    return state


def _interrupt_values(result):
    return [interrupt.value for interrupt in result.get("__interrupt__", ())]


def _run_to_completion(graph, state, config, resume_with="approved"):
    """Drive a run through however many gates it stops at, approving
    each. Returns (final_state, [gate ids stopped at, in order])."""
    result = graph.invoke(state, config)
    stops = []
    while result.get("__interrupt__"):
        values = _interrupt_values(result)
        stops.extend(value["gate"] for value in values)
        result = graph.invoke(Command(resume=resume_with), config)
    return result, stops


# --- topology ---------------------------------------------------------------


def test_every_pipeline_stage_is_a_node():
    graph = build_graph(build_in_memory_checkpointer())
    assert set(graph.get_graph().nodes) - {"__start__", "__end__"} == EXPECTED_NODES


def test_every_stub_stage_declares_the_branch_that_implements_it():
    assert set(nodes.IMPLEMENTED_BY) <= EXPECTED_NODES
    assert all(branch for branch in nodes.IMPLEMENTED_BY.values())


def test_stage_order_follows_the_documented_pipeline():
    graph = build_graph(build_in_memory_checkpointer())
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    for source, target in [
        ("__start__", "ingest"),
        ("classify_disclosure", "calibrate_rubrics"),
        ("calibrate_rubrics", "gate_rubric_signoff"),
        ("score_row", "gate_qualitative_disagreement"),
        ("gate_qualitative_disagreement", "apply_scoring_kernel"),
        ("apply_scoring_kernel", "block_capabilities"),
        ("block_capabilities", "build_profiles"),
        ("adjudicate_cluster", "gate_redundancy_verdict"),
        ("gate_redundancy_verdict", "apply_recommendation_policy"),
        ("apply_recommendation_policy", "detect_cost_outliers"),
        ("detect_cost_outliers", "explain_cost_outliers"),
        ("explain_cost_outliers", "gate_cost_outlier"),
        ("research_segment", "extract_and_ground_products"),
        ("extract_and_ground_products", "generate_narratives"),
        ("generate_narratives", "gate_narrative_grounding"),
        ("gate_narrative_grounding", "render_report"),
        ("render_report", "__end__"),
    ]:
        assert (source, target) in edges, f"missing edge {source} -> {target}"


# --- fan-out ----------------------------------------------------------------


def test_fan_out_produces_one_branch_per_subject():
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(), run_config("t-fanout"))

    assert set(final["disclosure"]) == {app["application_id"] for app in SYNTHETIC_APPLICATIONS}
    assert set(final["qualitative_scores"]) == {app["application_id"] for app in SYNTHETIC_APPLICATIONS}
    assert set(final["market_findings"]) == {seg["segment_id"] for seg in SYNTHETIC_SEGMENTS}
    assert final["branch_failures"] == []


def test_market_fan_out_is_per_segment_not_per_application():
    """CLAUDE.md section 8: a Scale-Tiered Overlap produces two
    differently-framed research targets from one cluster, so the market
    fan-out must key off segments, not rows or clusters."""
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(), run_config("t-segments"))

    assert len(final["market_findings"]) == len(SYNTHETIC_SEGMENTS)
    assert {"SEG-CLAIMS-HEAVY", "SEG-CLAIMS-LIGHT"} <= set(final["market_findings"])


def test_adjudication_branch_receives_only_its_own_clusters_profiles(monkeypatch):
    seen = {}

    def record(task):
        seen[task["cluster"]["cluster_id"]] = set(task["profiles"])
        return {}

    monkeypatch.setattr(nodes, "_stage_adjudication", record)
    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(), run_config("t-cluster-scope"))

    assert seen == {"CL-CLAIMS": {"SYN-001", "SYN-002"}, "CL-LEDGER": {"SYN-003"}}


def test_empty_portfolio_still_traverses_every_stage_and_gate():
    graph = build_graph(build_in_memory_checkpointer())
    state = initial_state(run_id="run-empty", applications=[], data_sensitivity="synthetic")
    final, stops = _run_to_completion(graph, state, run_config("t-empty"))

    visited = {entry["stage"] for entry in final["stage_log"]}
    fan_out_stages = {"classify_disclosure", "score_row", "adjudicate_cluster", "research_segment"}
    linear_stages = set(nodes.IMPLEMENTED_BY) | set(nodes.LANDED_BY)
    assert visited == linear_stages - fan_out_stages
    assert stops == [gates.GATE_RUBRIC_SIGNOFF]
    assert final["report"] == {}


# --- gates ------------------------------------------------------------------


def test_clean_run_stops_only_at_rubric_signoff():
    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(graph, _full_run_state(), run_config("t-clean"))

    assert stops == [gates.GATE_RUBRIC_SIGNOFF]
    assert final["rubric_signoff"] == {"signed_off": True, "decision": "approved"}
    assert final["rubrics"]["status"] == "signed_off"


def test_approving_gate_1_freezes_the_proposed_rubric_fields_unchanged():
    graph = build_graph(build_in_memory_checkpointer())
    config = run_config("t-approve-preserves-fields")
    graph.invoke(_full_run_state(), config)
    paused_fields = graph.get_state(config).values["rubrics"]["fields"]

    final = graph.invoke(Command(resume="approved"), config)

    assert final["rubrics"]["fields"] == paused_fields
    assert final["rubrics"]["status"] == "signed_off"


def test_rejecting_gate_1_freezes_status_to_rejected_but_keeps_the_proposal():
    graph = build_graph(build_in_memory_checkpointer())
    config = run_config("t-reject-rubrics")
    graph.invoke(_full_run_state(), config)
    paused_fields = graph.get_state(config).values["rubrics"]["fields"]

    final = graph.invoke(Command(resume={"action": "reject"}), config)

    assert final["rubric_signoff"] == {"signed_off": False, "decision": {"action": "reject"}}
    assert final["rubrics"]["status"] == "rejected"
    assert final["rubrics"]["fields"] == paused_fields


@pytest.mark.parametrize(
    "decision, approved",
    [
        ("approved", True),
        ({"action": "approve"}, True),
        ({"action": "approved"}, True),
        ({"signed_off": True}, True),
        ({"action": "reject"}, False),
        ({"action": "reject", "reason": "anchors look wrong"}, False),
        ({"signed_off_by": "internal-reviewer", "verdict": "rejected"}, False),
        ({"signed_off": False}, False),
        ("rejected", False),
        (None, False),
        ({}, False),
        ({"action": "APPROVE"}, False),  # case-sensitive: no silent normalization
    ],
)
def test_signoff_approval_parsing_fails_closed_on_anything_unrecognized(decision, approved):
    assert gates._is_signoff_approved(decision) is approved


def test_rubric_gate_blocks_before_any_row_is_scored():
    """Gate 1 is ordered ahead of the qualitative fan-out, not merely
    present: rubrics are frozen before a single row is scored."""
    graph = build_graph(build_in_memory_checkpointer())
    config = run_config("t-gate1-order")
    result = graph.invoke(_full_run_state(), config)

    assert _interrupt_values(result)[0]["gate"] == gates.GATE_RUBRIC_SIGNOFF
    paused = graph.get_state(config).values
    assert paused.get("qualitative_scores", {}) == {}


def test_signed_off_rubrics_do_not_stop_the_run_again():
    graph = build_graph(build_in_memory_checkpointer())
    state = _full_run_state(
        gate_decisions={gates.GATE_RUBRIC_SIGNOFF: {"decision": "approved", "item_count": 0}}
    )
    final, stops = _run_to_completion(graph, state, run_config("t-frozen-rubrics"))

    assert stops == []
    assert final["stage_log"]


@pytest.mark.parametrize(
    "gate",
    [
        gates.GATE_QUALITATIVE_DISAGREEMENT,
        gates.GATE_REDUNDANCY_VERDICT,
        gates.GATE_COST_OUTLIER,
        gates.GATE_NARRATIVE_GROUNDING,
    ],
)
def test_each_queue_driven_gate_fires_on_its_own_review_items(gate):
    graph = build_graph(build_in_memory_checkpointer())
    state = _full_run_state(
        review_queue=[{"gate": gate, "subject_id": "SYN-001", "reason": "synthetic fixture", "payload": {}}]
    )
    final, stops = _run_to_completion(graph, state, run_config(f"t-{gate}"))

    assert stops == [gates.GATE_RUBRIC_SIGNOFF, gate]
    assert final["gate_decisions"][gate] == {
        "decision": "approved",
        "reviewed_subject_ids": ["SYN-001"],
        "item_count": 1,
    }


def test_review_items_enqueued_by_a_fanned_out_branch_reach_their_gate(monkeypatch):
    """The realistic path: a worker branch decides mid-run that one of
    its fields needs review (score_row's real needs_review contract,
    CLAUDE.md section 7), and the gate downstream of the join picks it
    up."""

    def flag_ambiguous_row(task):
        if task["application"]["application_id"] != "SYN-002":
            return {}
        return {
            "business_criticality": {
                "field": "business_criticality",
                "raw_value": "x",
                "label": "medium",
                "points": 3,
                "confidence_label": "low",
                "source": "ensemble",
                "needs_review": True,
                "rationale": "synthetic fixture: ensemble range >= 2",
                "rubric_agreement": None,
                "ensemble_samples": None,
            }
        }

    monkeypatch.setattr(nodes, "_stage_qualitative", flag_ambiguous_row)
    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(graph, _full_run_state(), run_config("t-enqueued"))

    assert stops == [gates.GATE_RUBRIC_SIGNOFF, gates.GATE_QUALITATIVE_DISAGREEMENT]
    decision = final["gate_decisions"][gates.GATE_QUALITATIVE_DISAGREEMENT]
    assert decision["reviewed_subject_ids"] == ["SYN-002"]


def test_all_five_gates_are_reachable_in_one_run():
    graph = build_graph(build_in_memory_checkpointer())
    queue = [
        {"gate": gate, "subject_id": "SYN-001", "reason": "synthetic fixture", "payload": {}}
        for gate in gates.ALL_GATES
        if gate != gates.GATE_RUBRIC_SIGNOFF
    ]
    final, stops = _run_to_completion(graph, _full_run_state(review_queue=queue), run_config("t-all-gates"))

    assert stops == [
        gates.GATE_RUBRIC_SIGNOFF,
        gates.GATE_QUALITATIVE_DISAGREEMENT,
        gates.GATE_REDUNDANCY_VERDICT,
        gates.GATE_COST_OUTLIER,
        gates.GATE_NARRATIVE_GROUNDING,
    ]
    assert set(final["gate_decisions"]) == set(gates.ALL_GATES)


def test_reviewer_decision_is_recorded_verbatim():
    """A gate records what the reviewer actually returned -- a rejection
    is stored as a rejection, not normalized into an approval."""
    graph = build_graph(build_in_memory_checkpointer())
    config = run_config("t-rejection")
    graph.invoke(_full_run_state(), config)
    final = graph.invoke(Command(resume={"signed_off_by": "internal-reviewer", "verdict": "rejected"}), config)

    assert final["gate_decisions"][gates.GATE_RUBRIC_SIGNOFF]["decision"] == {
        "signed_off_by": "internal-reviewer",
        "verdict": "rejected",
    }


# --- resumability -----------------------------------------------------------


def test_suspended_run_resumes_through_a_freshly_compiled_graph():
    """State lives in the checkpointer, not in the compiled graph
    object -- which is what lets branch 16's poll/resume endpoint pick
    up a run it did not start."""
    checkpointer = build_in_memory_checkpointer()
    config = run_config("t-resume")

    first = build_graph(checkpointer).invoke(_full_run_state(), config)
    assert first.get("__interrupt__")

    second = build_graph(checkpointer)
    final, _ = _run_to_completion(second, Command(resume="approved"), config)

    assert set(final["market_findings"]) == {seg["segment_id"] for seg in SYNTHETIC_SEGMENTS}


def test_sqlite_checkpointer_survives_reopening_the_file(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    config = run_config("t-sqlite")

    saver = build_sqlite_checkpointer(db_path)
    build_graph(saver).invoke(_full_run_state(), config)
    saver.conn.close()

    assert db_path.exists()

    reopened = build_sqlite_checkpointer(db_path)
    final, _ = _run_to_completion(build_graph(reopened), Command(resume="approved"), config)
    reopened.conn.close()

    assert final["gate_decisions"][gates.GATE_RUBRIC_SIGNOFF]["decision"] == "approved"


def test_checkpoint_store_can_be_purged_on_the_bid_outcome_trigger(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    saver = build_sqlite_checkpointer(db_path)
    build_graph(saver).invoke(_full_run_state(), run_config("t-purge"))
    saver.conn.close()

    assert purge_checkpoint_store(db_path) is True
    assert not db_path.exists()
    assert purge_checkpoint_store(db_path) is False


def test_resumed_run_does_not_rerun_completed_stages():
    checkpointer = build_in_memory_checkpointer()
    graph = build_graph(checkpointer)
    config = run_config("t-no-rerun")

    graph.invoke(_full_run_state(), config)
    final, _ = _run_to_completion(graph, Command(resume="approved"), config)

    ingests = [entry for entry in final["stage_log"] if entry["stage"] == "ingest"]
    assert len(ingests) == 1


# --- parallel isolation -----------------------------------------------------


@pytest.mark.parametrize(
    "stage_attr, kind, failing_subject, surviving_key",
    [
        ("_stage_market_research", "market", "SEG-CLAIMS-LIGHT", "market_findings"),
        ("_stage_disclosure", "disclosure", "SYN-002", "disclosure"),
        ("_stage_qualitative", "qualitative", "SYN-002", "qualitative_scores"),
    ],
)
def test_one_failing_branch_does_not_take_down_its_siblings(
    monkeypatch, stage_attr, kind, failing_subject, surviving_key
):
    """CLAUDE.md section 8: branches are checkpointed independently so
    one branch's failure never fails the batch. The failure is recorded
    against its own subject and the run still completes."""

    def fail_one(task):
        subject = (
            task.get("segment", {}).get("segment_id")
            or task.get("application", {}).get("application_id")
        )
        if subject == failing_subject:
            raise RuntimeError("synthetic branch failure")
        return {}

    monkeypatch.setattr(nodes, stage_attr, fail_one)
    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(graph, _full_run_state(), run_config(f"t-isolation-{kind}"))

    assert stops == [gates.GATE_RUBRIC_SIGNOFF]
    assert final["branch_failures"] == [
        {
            "branch_kind": kind,
            "subject_id": failing_subject,
            "error": "RuntimeError: synthetic branch failure",
        }
    ]
    assert failing_subject not in final[surviving_key]
    assert len(final[surviving_key]) >= 2
    assert final["report"] == {}


def test_branch_results_merge_without_clobbering_each_other(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_stage_market_research",
        lambda task: {"segment_id": task["segment"]["segment_id"], "products": []},
    )
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(), run_config("t-merge"))

    assert all(
        finding["segment_id"] == segment_id for segment_id, finding in final["market_findings"].items()
    )


def test_reducers_are_order_independent():
    """The property the fan-out relies on: two branches' updates compose
    to the same state in either order, because each owns its own key."""
    left = {"SYN-001": {"ok": True}}
    right = {"SYN-002": {"ok": True}}
    assert merge_by_key(left, right) == merge_by_key(right, left)
    assert sorted(extend(["a"], ["b"])) == sorted(extend(["b"], ["a"]))
    assert merge_by_key(None, None) == {}
    assert extend(None, None) == []


# --- provider isolation -----------------------------------------------------


def test_only_the_disclosure_hook_may_reach_a_provider(monkeypatch):
    """Every stage except classify_disclosure is still a pass-through
    stub, and the autouse fixture above replaces classify_disclosure's
    own hook (_stage_disclosure) with a fake for this file -- so a full
    run touching app.llm.providers here would mean some OTHER node
    reached a provider on its own, bypassing its stage hook. That would
    be a real control-flow bug: nothing but a stage hook should ever be
    able to make a provider call. (The disclosure classifier's own real
    routing is verified in tests/test_disclosure_classifier.py.)"""
    from app.llm import providers

    def explode(*args, **kwargs):
        raise AssertionError("only a stage's own _stage_* hook may reach an LLM provider")

    monkeypatch.setattr(providers, "get_completion", explode)
    monkeypatch.setattr(providers, "GroqProvider", explode)
    monkeypatch.setattr(providers, "GeminiProvider", explode)

    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(), run_config("t-no-llm"))

    assert final["stage_log"]


def test_classify_disclosure_delegates_to_the_real_classifier(monkeypatch):
    """Verifies the wiring this branch adds -- that the stage hook calls
    app.disclosure.classifier.classify_row with the row's own data and
    the run's sensitivity flag -- without exercising the classifier's own
    LLM-calling internals (that belongs to its own test module). Restores
    the real _stage_disclosure over this file's blanket autouse fake for
    this one test, then substitutes classify_row itself so the assertion
    below never reaches a real provider."""
    from app.disclosure import classifier as disclosure_classifier

    monkeypatch.setattr(nodes, "_stage_disclosure", _REAL_STAGE_DISCLOSURE)

    calls = []

    def fake_classify_row(application, *, application_id, data_sensitivity):
        calls.append((application_id, data_sensitivity))
        return {}

    monkeypatch.setattr(disclosure_classifier, "classify_row", fake_classify_row)

    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(), run_config("t-real-wiring"))

    from app.llm.providers import DataSensitivity

    assert sorted(calls) == [
        (app["application_id"], DataSensitivity.SYNTHETIC) for app in SYNTHETIC_APPLICATIONS
    ]


def test_calibrate_rubrics_delegates_to_the_real_module(monkeypatch):
    """Verifies the wiring this branch adds -- that the node hands the
    disclosure-gated applications and the run's sensitivity flag to
    app.rubric.calibration.calibrate_rubrics -- without exercising that
    module's own LLM-calling internals (its own test module's job).
    Restores the real calibrate_rubrics over this file's blanket autouse
    fake, then substitutes the module-level function it calls so the
    assertion below never reaches a real provider."""
    monkeypatch.setattr(nodes, "_stage_disclosure", _REAL_STAGE_DISCLOSURE)
    monkeypatch.setattr(nodes, "calibrate_rubrics", _REAL_CALIBRATE_RUBRICS)

    calls = []

    def fake_calibrate(gated_applications, *, data_sensitivity):
        calls.append((tuple(sorted(app["application_id"] for app in gated_applications)), data_sensitivity))
        return {}

    monkeypatch.setattr(rubric_calibration, "calibrate_rubrics", fake_calibrate)

    applications = [
        {"application_id": "SYN-001", "business_criticality": "Strategic"},
        {"application_id": "SYN-002", "business_criticality": "Supports core processes"},
    ]
    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(applications=applications), run_config("t-rubric-wiring"))

    from app.llm.providers import DataSensitivity

    assert calls == [(("SYN-001", "SYN-002"), DataSensitivity.SYNTHETIC)]


def test_calibrate_rubrics_skips_a_row_whose_disclosure_branch_failed(monkeypatch):
    """A row absent from state["disclosure"] (its branch failed --
    branch_failures, not a result) must not fall back to its raw ungated
    value -- there is no confirmation it was actually answered."""
    monkeypatch.setattr(nodes, "calibrate_rubrics", _REAL_CALIBRATE_RUBRICS)

    def fail_for_syn_002(task):
        if task["application"]["application_id"] == "SYN-002":
            raise RuntimeError("synthetic branch failure")
        return {
            "results": {},
            "gated_application": dict(task["application"]),
            "phase2_agenda": [],
        }

    monkeypatch.setattr(nodes, "_stage_disclosure", fail_for_syn_002)

    calls = []

    def fake_calibrate(gated_applications, *, data_sensitivity):
        calls.append(list(gated_applications))
        return {}

    monkeypatch.setattr(rubric_calibration, "calibrate_rubrics", fake_calibrate)

    applications = [
        {"application_id": "SYN-001", "business_criticality": "Strategic"},
        {"application_id": "SYN-002", "business_criticality": "Supports core processes"},
    ]
    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(applications=applications), run_config("t-rubric-partial-disclosure"))

    assert calls == [[{"application_id": "SYN-001", "business_criticality": "Strategic"}]]


def test_score_row_delegates_to_the_real_scorer(monkeypatch):
    """Verifies the wiring this branch adds -- that the stage hook calls
    app.qualitative_scoring.scorer.score_row with the row's
    disclosure-gated data, the run's rubrics, and its sensitivity flag --
    without exercising the scorer's own LLM-calling internals (its own
    test module's job)."""
    from app.qualitative_scoring import scorer as qualitative_scorer

    monkeypatch.setattr(nodes, "_stage_qualitative", _REAL_STAGE_QUALITATIVE)

    calls = []

    def fake_score_row(gated_application, rubrics, *, application_id, data_sensitivity):
        calls.append((application_id, gated_application.get("application_id"), rubrics, data_sensitivity))
        return {}

    monkeypatch.setattr(qualitative_scorer, "score_row", fake_score_row)

    applications = [{"application_id": "SYN-001"}, {"application_id": "SYN-002"}]
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(
        graph, _full_run_state(applications=applications), run_config("t-score-row-wiring")
    )

    from app.llm.providers import DataSensitivity

    # The autouse fake calibrate_rubrics proposes {"status": "proposed", ...},
    # and approving gate 1 (the default resume) freezes it to signed_off --
    # this is the same `rubrics` every score_row call receives.
    signed_off_rubrics = final["rubrics"]
    assert signed_off_rubrics["status"] == "signed_off"
    assert sorted(calls) == [
        ("SYN-001", "SYN-001", signed_off_rubrics, DataSensitivity.SYNTHETIC),
        ("SYN-002", "SYN-002", signed_off_rubrics, DataSensitivity.SYNTHETIC),
    ]


def test_score_row_only_fans_out_to_rows_with_a_disclosure_result(monkeypatch):
    """A row whose disclosure branch failed has no gated_application to
    give the scorer -- it must not fall back to the raw ungated value."""
    from app.qualitative_scoring import scorer as qualitative_scorer

    monkeypatch.setattr(nodes, "_stage_qualitative", _REAL_STAGE_QUALITATIVE)

    def fail_for_syn_002(task):
        if task["application"]["application_id"] == "SYN-002":
            raise RuntimeError("synthetic branch failure")
        return _fake_stage_disclosure(task)

    monkeypatch.setattr(nodes, "_stage_disclosure", fail_for_syn_002)

    calls = []
    monkeypatch.setattr(
        qualitative_scorer, "score_row",
        lambda gated_application, rubrics, *, application_id, data_sensitivity: calls.append(application_id) or {},
    )

    applications = [{"application_id": "SYN-001"}, {"application_id": "SYN-002"}]
    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(applications=applications), run_config("t-score-row-partial"))

    assert calls == ["SYN-001"]


def test_apply_scoring_kernel_reads_gated_costs_and_resolved_qualitative_labels():
    """Verifies the wiring this branch adds: apply_scoring_kernel builds
    kernel.ScoringInput from the disclosure-gated application (identity
    and cost fields) and the qualitative scorer's resolved labels (TIM-E
    axes) -- the exact scoring arithmetic is
    tests/test_scoring_kernel.py's job, not this one's."""
    scored_axes = [
        "business_criticality", "strategic_relevance", "business_fitness", "usage_adoption",
        "application_stability", "maintainability", "availability", "reliability", "scalability",
        "functional_redundancy",
    ]

    def fake_stage_qualitative(task):
        return {
            field: {
                "field": field, "raw_value": "x", "label": "very high", "points": 5,
                "confidence_label": "high", "source": "single_call", "needs_review": False,
                "rationale": "r", "rubric_agreement": True, "ensemble_samples": None,
            }
            for field in scored_axes
        }

    applications = [
        {
            "application_id": "SYN-001",
            "application_name": "Synthetic App",
            "business_capability_l2": "L2",
            "business_capability_l3": "L3",
            "annual_fte_cost": 100000,
            "annual_license_cost": 50000,
            "annual_infrastructure_cost": 20000,
            "other_costs": 1000,
        }
    ]

    graph = build_graph(build_in_memory_checkpointer())
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(nodes, "_stage_qualitative", fake_stage_qualitative)
        final, _ = _run_to_completion(
            graph, _full_run_state(applications=applications), run_config("t-kernel-wiring")
        )

    result = final["kernel_results"]["SYN-001"]
    assert result["tim_e_score"] is not None
    assert result["tim_e_decision"] == "Invest"
    assert result["floor_applied"] is None

def test_block_capabilities_delegates_to_the_real_module(monkeypatch):
    """Verifies the wiring this branch adds: block_capabilities calls
    app.redundancy.blocking.block_by_capability over state["applications"]
    -- deterministic, so no provider mocking is needed here at all."""
    from app.redundancy import blocking

    monkeypatch.setattr(nodes, "block_capabilities", _REAL_BLOCK_CAPABILITIES)

    applications = [
        {"application_id": "SYN-001", "business_capability_l1": "Finance", "business_capability_l2": "R2R"},
        {"application_id": "SYN-002", "business_capability_l1": "Finance", "business_capability_l2": "R2R"},
    ]
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(applications=applications), run_config("t-block-wiring"))

    assert len(final["clusters"]) == 1
    assert set(final["clusters"][0]["application_ids"]) == {"SYN-001", "SYN-002"}


def test_build_profiles_delegates_to_the_real_module_and_prefers_gated_data(monkeypatch):
    """Verifies the wiring this branch adds: build_profiles prefers each
    row's disclosure-gated application over the raw one when a
    disclosure result exists for it. classify_disclosure runs for real
    in this test (its output is what build_profiles must prefer), so
    _stage_disclosure is overridden directly rather than pre-seeding
    state["disclosure"] -- the real classify_disclosure fan-out would
    otherwise overwrite a pre-seeded value for the same application on
    its own first execution."""
    monkeypatch.setattr(nodes, "build_profiles", _REAL_BUILD_PROFILES)
    monkeypatch.setattr(
        nodes, "_stage_disclosure",
        lambda task: {"gated_application": {"application_id": "SYN-001", "business_criticality": None}},
    )

    applications = [{"application_id": "SYN-001", "business_criticality": "Strategic"}]
    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(applications=applications), run_config("t-profiles-wiring"))

    assert final["profiles"]["SYN-001"]["scale_usage"]["business_criticality"] is None


def test_stage_adjudication_delegates_to_the_real_adjudicator_and_policy(monkeypatch):
    """Verifies the wiring this branch adds: _stage_adjudication builds
    ApplicationProfile objects from the task's serialized profiles,
    calls the real adjudicator and recommendation_policy, and folds the
    recommendation into each verdict -- without exercising either
    module's own LLM-calling internals (their own test modules' job)."""
    from app.redundancy import adjudicator, profile_builder

    monkeypatch.setattr(nodes, "_stage_adjudication", _REAL_STAGE_ADJUDICATION)

    def fake_adjudicate_cluster(cluster_id, profiles, *, data_sensitivity):
        assert cluster_id == "CL-CLAIMS"
        assert {p.application_id for p in profiles} == {"SYN-001", "SYN-002"}
        return [
            adjudicator.AdjudicationVerdict(
                cluster_id=cluster_id, application_id_a="SYN-001", application_id_b="SYN-002",
                typology=adjudicator.TRUE_DUPLICATE, resolution="unanimous", votes=[],
                mandatory_review=True, rationale="synthetic wiring test",
            )
        ]

    monkeypatch.setattr(adjudicator, "adjudicate_cluster", fake_adjudicate_cluster)

    # SYNTHETIC_PROFILES (the default from _full_run_state) is a
    # placeholder shape, not real ApplicationProfile.as_dict() output --
    # supply properly-shaped profiles for CL-CLAIMS's members so
    # ApplicationProfile.from_dict can actually deserialize them.
    real_profiles = {
        application_id: profile_builder.build_profile({"application_id": application_id}).as_dict()
        for application_id in ("SYN-001", "SYN-002", "SYN-003")
    }
    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(
        graph, _full_run_state(profiles=real_profiles), run_config("t-adjudication-wiring")
    )

    assert gates.GATE_REDUNDANCY_VERDICT in stops
    verdict = final["verdicts"][0]
    assert verdict["typology"] == adjudicator.TRUE_DUPLICATE
    assert verdict["recommendation"]["recommendation"]  # non-empty -- recommendation_policy actually ran
    assert verdict["recommendation"]["mandatory_review"] is True


def test_gate_3_fires_on_the_recommendations_own_review_flag_not_only_the_verdicts(monkeypatch):
    """The specific property _stage_adjudication exists to make possible
    (see its docstring): a Scale-Tiered Overlap verdict the ensemble
    itself did not flag (mandatory_review=False) must still reach gate 3
    once recommendation_policy resolves it to a consolidation
    recommendation -- proving gate 3 is wired off the *combined* decision,
    not off the ensemble's confidence alone."""
    from app.redundancy import adjudicator, profile_builder

    monkeypatch.setattr(nodes, "_stage_adjudication", _REAL_STAGE_ADJUDICATION)

    def fake_adjudicate_cluster(cluster_id, profiles, *, data_sensitivity):
        return [
            adjudicator.AdjudicationVerdict(
                cluster_id=cluster_id, application_id_a="SYN-001", application_id_b="SYN-002",
                typology=adjudicator.SCALE_TIERED_OVERLAP, resolution="unanimous", votes=[],
                mandatory_review=False,  # the ensemble itself saw no reason to flag this
                rationale="synthetic wiring test",
            )
        ]

    monkeypatch.setattr(adjudicator, "adjudicate_cluster", fake_adjudicate_cluster)

    # A heavier, cheaper-per-FTE platform and a lighter, pricier-per-FTE
    # candidate, both matching on every gate -- nothing blocks
    # consolidation, so recommendation_policy resolves to "migrate."
    heavy = profile_builder.build_profile({
        "application_id": "SYN-001", "fte_count": 50, "business_criticality": "low",
        "application_stability": "very high", "application_security_level": "Confidential",
        "annual_fte_cost": 500_000,
    }).as_dict()
    light = profile_builder.build_profile({
        "application_id": "SYN-002", "fte_count": 2, "business_criticality": "low",
        "application_security_level": "Confidential", "annual_fte_cost": 100_000,
    }).as_dict()

    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(
        graph, _full_run_state(profiles={"SYN-001": heavy, "SYN-002": light, "SYN-003": light}),
        run_config("t-gate3-combined-flag"),
    )

    assert gates.GATE_REDUNDANCY_VERDICT in stops
    verdict = final["verdicts"][0]
    assert "Migrate" in verdict["recommendation"]["recommendation"]
    assert verdict["recommendation"]["mandatory_review"] is True


def test_detect_cost_outliers_delegates_to_the_real_module(monkeypatch):
    """Verifies the wiring this branch adds: detect_cost_outliers builds
    ApplicationProfile objects from state["profiles"] and calls
    app.cost_intelligence.outlier_detection.detect_cost_outliers over
    state["clusters"] -- deterministic, so no provider mocking is needed
    here at all."""
    from app.redundancy import profile_builder

    monkeypatch.setattr(nodes, "detect_cost_outliers", _REAL_DETECT_COST_OUTLIERS)

    profiles = {
        f"SYN-{i:03d}": profile_builder.build_profile(
            {"application_id": f"SYN-{i:03d}", "fte_count": 1, "annual_fte_cost": cost}
        ).as_dict()
        for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 500_000], start=1)
    }
    clusters = [{"cluster_id": "CL-CLAIMS", "application_ids": list(profiles)}]

    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(
        graph, _full_run_state(clusters=clusters, profiles=profiles), run_config("t-cost-outlier-wiring")
    )

    assert len(final["cost_outliers"]) == 1
    assert final["cost_outliers"][0]["application_id"] == "SYN-005"


def test_explain_cost_outliers_delegates_to_the_real_module_and_wires_gate_4(monkeypatch):
    """Verifies the wiring this branch adds: explain_cost_outliers calls
    app.cost_intelligence.explainability.explain_outliers over
    state["cost_outliers"], and a low-confidence verdict reaches gate 4
    -- without exercising explainability's own LLM-calling internals
    (its own test module's job)."""
    from app.cost_intelligence import explainability
    from app.redundancy import profile_builder

    monkeypatch.setattr(nodes, "detect_cost_outliers", _REAL_DETECT_COST_OUTLIERS)
    monkeypatch.setattr(nodes, "explain_cost_outliers", _REAL_EXPLAIN_COST_OUTLIERS)

    def fake_explain_outliers(flags, profiles, *, data_sensitivity):
        return [
            {**flag.as_dict(), "explainability": {
                "explainable": None, "confidence": None,
                "rationale": "synthetic wiring test", "needs_review": True,
            }}
            for flag in flags
        ]

    monkeypatch.setattr(explainability, "explain_outliers", fake_explain_outliers)

    profiles = {
        f"SYN-{i:03d}": profile_builder.build_profile(
            {"application_id": f"SYN-{i:03d}", "fte_count": 1, "annual_fte_cost": cost}
        ).as_dict()
        for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 500_000], start=1)
    }
    clusters = [{"cluster_id": "CL-CLAIMS", "application_ids": list(profiles)}]

    graph = build_graph(build_in_memory_checkpointer())
    final, stops = _run_to_completion(
        graph, _full_run_state(clusters=clusters, profiles=profiles), run_config("t-explain-wiring")
    )

    assert gates.GATE_COST_OUTLIER in stops
    assert final["cost_outliers"][0]["explainability"]["needs_review"] is True


def test_run_input_defaults_to_real_data_sensitivity():
    """Fails closed: an omitted flag is treated as real client data, so
    the Gemini fallback stays unavailable unless a caller declares
    synthetic fixtures (CLAUDE.md section 11)."""
    assert initial_state("run-x")["data_sensitivity"] == "real"


def test_every_fanned_out_branch_receives_the_runs_sensitivity_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(
        nodes, "_stage_qualitative", lambda task: seen.append(task["data_sensitivity"]) or {}
    )
    graph = build_graph(build_in_memory_checkpointer())
    _run_to_completion(graph, _full_run_state(), run_config("t-sensitivity"))

    assert seen == ["synthetic"] * len(SYNTHETIC_APPLICATIONS)
