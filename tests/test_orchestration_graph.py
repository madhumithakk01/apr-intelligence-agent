"""Orchestration graph tests -- topology, resumability, gates, isolation.

The graph is all control flow and no stage behavior on this branch, so
these tests assert exactly the control-flow properties every later
branch will rely on and nothing about what a stage returns:

  * every pipeline stage in CLAUDE.md section 5 is a node, in order
  * a run suspends at a gate and resumes from the checkpointer alone --
    including through a freshly compiled graph and a reopened SQLite
    file, which is what proves the state lives in the checkpoint rather
    than in the process
  * all five gates fire, each only on its own trigger
  * one fanned-out branch failing leaves its siblings' results intact
  * nothing in the graph reaches an LLM provider on this branch
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
from synthetic_fixtures import (
    SYNTHETIC_APPLICATIONS,
    SYNTHETIC_CLUSTERS,
    SYNTHETIC_PROFILES,
    SYNTHETIC_SEGMENTS,
)

_REAL_BLOCK_CAPABILITIES = nodes.block_capabilities
_REAL_BUILD_PROFILES = nodes.build_profiles
_REAL_STAGE_ADJUDICATION = nodes._stage_adjudication
"""Captured at import time, before the autouse fixture below ever runs,
so a test that wants the real wiring back can restore it explicitly."""


def _fake_block_capabilities(state):
    return {"clusters": list(state.get("clusters") or []), "stage_log": [nodes._stage("block_capabilities")]}


def _fake_build_profiles(state):
    return {"stage_log": [nodes._stage("build_profiles")]}


def _fake_stage_adjudication(task):
    return {}


@pytest.fixture(autouse=True)
def _no_real_llm_calls_in_this_file(monkeypatch):
    """This file tests orchestration control flow, not what
    block_capabilities/build_profiles/adjudicate_cluster themselves
    compute (see tests/test_blocking.py, tests/test_profile_builder.py,
    tests/test_redundancy_adjudicator.py, and
    tests/test_recommendation_policy.py). Autouse so no test here can
    accidentally reach a real provider via the adjudicator -- including
    on a machine that happens to have a real GROQ_API_KEY exported --
    and so the rest of this file's tests can keep pre-setting
    state["clusters"]/state["profiles"] directly (_full_run_state below)
    without block_capabilities/build_profiles silently recomputing and
    overwriting them from state["applications"]. A test that cares about
    the real wiring restores the captured original above, which simply
    shadows this fixture's patch for that one test."""
    monkeypatch.setattr(nodes, "block_capabilities", _fake_block_capabilities)
    monkeypatch.setattr(nodes, "build_profiles", _fake_build_profiles)
    monkeypatch.setattr(nodes, "_stage_adjudication", _fake_stage_adjudication)


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
    """The realistic path: a worker branch decides mid-run that its row
    needs review, and the gate downstream of the join picks it up."""

    def flag_ambiguous_row(task):
        if task["application"]["application_id"] != "SYN-002":
            return {}
        return {
            "review_items": [
                {
                    "gate": gates.GATE_QUALITATIVE_DISAGREEMENT,
                    "subject_id": "SYN-002",
                    "reason": "synthetic fixture: ensemble range >= 2",
                    "payload": {},
                }
            ]
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


def test_no_stage_reaches_an_llm_provider_on_this_branch(monkeypatch):
    """Control flow only: every stage is a stub, so a full run must not
    touch app.llm.providers. This test is what makes it obvious when a
    later branch wires a real call in -- it fails loudly rather than
    letting an unreviewed provider call slip into the skeleton."""
    from app.llm import providers

    def explode(*args, **kwargs):
        raise AssertionError("orchestration skeleton must not call an LLM provider")

    monkeypatch.setattr(providers, "get_completion", explode)
    monkeypatch.setattr(providers, "GroqProvider", explode)
    monkeypatch.setattr(providers, "GeminiProvider", explode)

    graph = build_graph(build_in_memory_checkpointer())
    final, _ = _run_to_completion(graph, _full_run_state(), run_config("t-no-llm"))

    assert final["stage_log"]


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
