"""Pipeline stage nodes for the orchestration graph.

Most nodes here are still deliberate stubs (IMPLEMENTED_BY records which
future branch replaces each one, and each stub is honest about it rather
than returning a plausible-looking fake result -- a stub that invented
an output would let a later branch's tests pass for the wrong reason).
``block_capabilities``, ``build_profiles``, ``adjudicate_cluster``, and
``apply_recommendation_policy`` are real as of feat/redundancy-adjudicator
(branch 10) -- the first two were orphaned by feat/redundancy-blocking-
profile (branch 9), which built the modules they call but, based strictly
on refactor/scoring-kernel-consolidation (branch 3), never had this graph
in its own ancestry to wire them into. LANDED_BY records the transition
to real the same way IMPLEMENTED_BY records a pending one, so the stage
log always says which is which.

Fan-out workers (``classify_disclosure``, ``score_row``,
``adjudicate_cluster``, ``research_segment``) never raise past their own
branch: a failure is recorded in ``branch_failures`` for that subject and
the remaining branches finish normally (CLAUDE.md section 8). Each
delegates its body to a ``_stage_*`` function purely so that isolation
is testable by substituting one. ``block_capabilities`` and
``build_profiles`` are linear nodes, not fanned out, and have no such
split -- each has exactly one call site (a module-level function
elsewhere) for a test to substitute.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.llm.providers import DataSensitivity
from app.orchestration import gates
from app.orchestration.state import (
    ClusterTask,
    GraphState,
    RowTask,
    SegmentTask,
    StageStatus,
)
from app.redundancy import adjudicator, blocking
from app.redundancy import profile_builder as profile_builder_module
from app.redundancy import recommendation_policy

IMPLEMENTED_BY = {
    "ingest": "fix/ingestion-integrity (2, landed -- wiring deferred to branch 6)",
    "classify_disclosure": "feat/disclosure-classifier (6)",
    "calibrate_rubrics": "feat/rubric-calibration (7)",
    "score_row": "feat/qualitative-scoring (8)",
    "apply_scoring_kernel": "refactor/scoring-kernel-consolidation (3, landed -- wiring deferred to branch 8)",
    "detect_cost_outliers": "feat/cost-outlier-detection (11)",
    "explain_cost_outliers": "feat/cost-outlier-detection (11)",
    "research_segment": "feat/market-intelligence-agent (12)",
    "extract_and_ground_products": "feat/product-extraction-grounding (13)",
    "generate_narratives": "feat/narrative-generation (14)",
    "render_report": "feat/report-rendering-consolidation (15)",
}

LANDED_BY = {
    "block_capabilities": "feat/redundancy-blocking-profile (9) + feat/redundancy-adjudicator (10, wiring)",
    "build_profiles": "feat/redundancy-blocking-profile (9) + feat/redundancy-adjudicator (10, wiring)",
    "adjudicate_cluster": "feat/redundancy-adjudicator (10)",
    "apply_recommendation_policy": "feat/redundancy-adjudicator (10)",
}


def _stage(name: str) -> StageStatus:
    if name in LANDED_BY:
        return StageStatus(stage=name, status="complete", implemented_by_branch=LANDED_BY[name])
    return StageStatus(stage=name, status="stub", implemented_by_branch=IMPLEMENTED_BY[name])


def _failure(kind: str, subject_id: str, error: BaseException) -> Dict[str, Any]:
    return {"branch_kind": kind, "subject_id": subject_id, "error": f"{type(error).__name__}: {error}"}


def _data_sensitivity(source: Dict[str, Any]) -> DataSensitivity:
    """Fails closed: an unrecognized or missing flag is treated as real
    client data, matching graph.initial_state's default (CLAUDE.md
    section 11) -- a run can only reach Gemini by explicitly declaring
    itself synthetic, never by a blank or malformed flag. `source` is
    GraphState for a linear node or a *Task dict for a fanned-out
    worker; both carry the same "data_sensitivity" key."""
    if source.get("data_sensitivity") == "synthetic":
        return DataSensitivity.SYNTHETIC
    return DataSensitivity.REAL


# --- linear deterministic stages -------------------------------------------


def ingest(state: GraphState) -> Dict[str, Any]:
    """Pass-through. The real loader/validator/dedup path already exists
    in app/ingestion (branch 2); it is wired in at branch 6, once the
    disclosure classifier gives its output somewhere to go. Applications
    are supplied in the run input state until then."""
    return {"applications": list(state.get("applications") or []), "stage_log": [_stage("ingest")]}


def calibrate_rubrics(state: GraphState) -> Dict[str, Any]:
    """Single structured LLM call, once per field per engagement
    (CLAUDE.md section 5), frozen after gate 1 sign-off."""
    return {"rubrics": dict(state.get("rubrics") or {}), "stage_log": [_stage("calibrate_rubrics")]}


def apply_scoring_kernel(state: GraphState) -> Dict[str, Any]:
    """Deterministic TIM-E / COTS-fit scoring. app/scoring/kernel.py is
    the landed implementation, but it consumes calibrated qualitative
    scores that do not exist until branch 8 -- so its call site lands
    there rather than feeding the kernel this branch's stub output."""
    return {"stage_log": [_stage("apply_scoring_kernel")]}


def block_capabilities(state: GraphState) -> Dict[str, Any]:
    """Deterministic, generous capability blocking
    (app.redundancy.blocking, branch 9). Coarse on purpose -- a pairing
    missed here can never be recovered downstream (section 5). Reads the
    raw ingested applications directly, never disclosure-gated: capability
    tags are the blocking key and are "rarely withheld... block
    generously" (section 9), unlike the comparison fields build_profiles
    treats as unknown once disclosure has gated them out."""
    clusters = blocking.block_by_capability(state.get("applications") or [])
    return {
        "clusters": [cluster.as_dict() for cluster in clusters],
        "stage_log": [_stage("block_capabilities")],
    }


def build_profiles(state: GraphState) -> Dict[str, Any]:
    """Multi-axis profile building (app.redundancy.profile_builder,
    branch 9) -- five axes kept separate, never pre-blended into a single
    similarity score (section 9). Prefers each application's disclosure-
    gated data when a disclosure result exists for it (branch 6): a
    withheld comparison field must show up as unknown in the profile,
    never as its raw value. Falls back to the raw ingested application
    for a row whose disclosure branch never ran -- there is nothing to
    gate against yet, not license to treat every field as confirmed."""
    disclosure = state.get("disclosure") or {}
    applications = []
    for application in state.get("applications") or []:
        application_id = application.get("application_id")
        entry = disclosure.get(application_id) if application_id else None
        gated = entry.get("gated_application") if entry else None
        applications.append(gated if gated is not None else application)

    profiles = profile_builder_module.build_profiles(applications)
    return {
        "profiles": {application_id: profile.as_dict() for application_id, profile in profiles.items()},
        "stage_log": [_stage("build_profiles")],
    }


def apply_recommendation_policy(state: GraphState) -> Dict[str, Any]:
    """Deterministic, non-compensatory, ordered policy (section 9):
    classification gate, then criticality ceiling, then normalized cost,
    then technical feasibility. The policy itself already ran per pair
    inside adjudicate_cluster's fan-out (see that node's docstring for
    why gate 3's review trigger needs it to) -- this linear node just
    assembles the final recommendations list from what state["verdicts"]
    already carries."""
    recommendations = [
        verdict["recommendation"]
        for verdict in (state.get("verdicts") or [])
        if "recommendation" in verdict
    ]
    return {"recommendations": recommendations, "stage_log": [_stage("apply_recommendation_policy")]}


def detect_cost_outliers(state: GraphState) -> Dict[str, Any]:
    """Deterministic statistics decide the flag; the minimum peer cluster
    size floor lives in governance_params (section 12)."""
    return {
        "cost_outliers": list(state.get("cost_outliers") or []),
        "stage_log": [_stage("detect_cost_outliers")],
    }


def explain_cost_outliers(state: GraphState) -> Dict[str, Any]:
    """Single LLM call judging only whether an already-flagged outlier is
    explainable -- it never decides the flag itself."""
    return {"stage_log": [_stage("explain_cost_outliers")]}


def extract_and_ground_products(state: GraphState) -> Dict[str, Any]:
    """Single structured extraction call plus a deterministic
    claim-level grounding check -- every individual claim, not just the
    product name (section 5)."""
    return {"stage_log": [_stage("extract_and_ground_products")]}


def generate_narratives(state: GraphState) -> Dict[str, Any]:
    """Bounded retry-once generation with a scripted fallback to
    structured bullets. The stopping rule is fixed in code, never model
    judgment (section 5) -- which is why this is a node, not a loop."""
    return {"stage_log": [_stage("generate_narratives")]}


def render_report(state: GraphState) -> Dict[str, Any]:
    """Deterministic rendering, into the single consolidated renderer
    that replaces the current duplicate report_service/report_renderer
    pair."""
    return {"report": dict(state.get("report") or {}), "stage_log": [_stage("render_report")]}


# --- fanned-out branch bodies ----------------------------------------------
# Separated from their worker wrappers so a test can substitute one and
# assert that the surviving branches still complete.


def _stage_disclosure(task: RowTask) -> Dict[str, Any]:
    return {}


def _stage_qualitative(task: RowTask) -> Dict[str, Any]:
    return {}


def _stage_adjudication(task: ClusterTask) -> Dict[str, Any]:
    """CLAUDE.md sections 9 and 10: every pair within this cluster is
    adjudicated (app.redundancy.adjudicator, an ensemble, not an agent),
    then the deterministic recommendation policy
    (app.redundancy.recommendation_policy) is applied to each verdict
    right here, in the same fanned-out branch. Both must run before gate
    3's review trigger can be correctly decided -- a Scale-Tiered
    Overlap recommending consolidation routes to review regardless of
    the ensemble's own confidence, and that only becomes knowable once
    the policy has run. Doing both here, rather than splitting the
    policy into its own linear node ahead of the gate, avoids
    restructuring the graph's existing topology (gate 3 already sits
    immediately after this fan-out joins)."""
    cluster = task.get("cluster") or {}
    cluster_id = str(cluster.get("cluster_id") or "unknown")
    profiles_data = task.get("profiles") or {}
    application_ids = cluster.get("application_ids") or []

    profiles = [
        profile_builder_module.ApplicationProfile.from_dict(profiles_data[application_id])
        for application_id in application_ids
        if application_id in profiles_data
    ]
    profiles_by_id = {profile.application_id: profile for profile in profiles}

    verdicts = adjudicator.adjudicate_cluster(
        cluster_id, profiles, data_sensitivity=_data_sensitivity(task)
    )

    entries: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []
    for verdict in verdicts:
        profile_a = profiles_by_id.get(verdict.application_id_a)
        profile_b = profiles_by_id.get(verdict.application_id_b)
        recommendation = recommendation_policy.evaluate(verdict, profile_a, profile_b)
        entry = {**verdict.as_dict(), "recommendation": recommendation.as_dict()}
        entries.append(entry)
        if recommendation.mandatory_review:
            review_items.append(
                {
                    "gate": gates.GATE_REDUNDANCY_VERDICT,
                    "subject_id": f"{cluster_id}:{verdict.application_id_a}:{verdict.application_id_b}",
                    "reason": f"{verdict.typology}: {recommendation.recommendation}",
                    "payload": entry,
                }
            )

    return {"verdicts": entries, "review_items": review_items}


def _stage_market_research(task: SegmentTask) -> Dict[str, Any]:
    return {}


def _row_id(task: RowTask) -> str:
    return str((task.get("application") or {}).get("application_id") or "unknown")


def classify_disclosure(task: RowTask) -> Dict[str, Any]:
    """One fanned-out branch per application row (section 6)."""
    subject = _row_id(task)
    try:
        result = _stage_disclosure(task)
    except Exception as exc:  # branch-local: never fails the batch
        return {
            "branch_failures": [_failure("disclosure", subject, exc)],
            "stage_log": [_stage("classify_disclosure")],
        }
    return {"disclosure": {subject: result}, "stage_log": [_stage("classify_disclosure")]}


def score_row(task: RowTask) -> Dict[str, Any]:
    """One fanned-out branch per application row. Single call by default,
    escalating to a 3-sample ensemble on low confidence (section 7); an
    ensemble range of >= 2 enqueues a gate 2 review item."""
    subject = _row_id(task)
    try:
        result = _stage_qualitative(task)
    except Exception as exc:
        return {
            "branch_failures": [_failure("qualitative", subject, exc)],
            "stage_log": [_stage("score_row")],
        }
    update: Dict[str, Any] = {
        "qualitative_scores": {subject: result},
        "stage_log": [_stage("score_row")],
    }
    review = result.get("review_items")
    if review:
        update["review_queue"] = list(review)
    return update


def adjudicate_cluster(task: ClusterTask) -> Dict[str, Any]:
    """One fanned-out branch per candidate cluster -- possibly several
    pairwise verdicts per cluster (O(k^2) within a cluster, CLAUDE.md
    section 9's accepted scaling cost). 3-sample ensemble into the
    five-way typology per pair -- an ensemble, not an agent. Gate 3
    review items are enqueued per pair, not per cluster."""
    subject = str((task.get("cluster") or {}).get("cluster_id") or "unknown")
    try:
        result = _stage_adjudication(task)
    except Exception as exc:
        return {
            "branch_failures": [_failure("redundancy", subject, exc)],
            "stage_log": [_stage("adjudicate_cluster")],
        }
    update: Dict[str, Any] = {"stage_log": [_stage("adjudicate_cluster")]}
    verdicts = result.get("verdicts")
    if verdicts:
        update["verdicts"] = list(verdicts)
    review_items = result.get("review_items")
    if review_items:
        update["review_queue"] = list(review_items)
    return update


def research_segment(task: SegmentTask) -> Dict[str, Any]:
    """One fanned-out branch per redundancy-surviving segment -- the only
    genuine agent in the system (section 8). Its internal search loop is
    a subgraph built on branch 12; this node is the fan-out slot that
    subgraph plugs into, checkpointed per branch so one branch's API
    failure resumes independently of the others."""
    subject = str((task.get("segment") or {}).get("segment_id") or "unknown")
    try:
        result = _stage_market_research(task)
    except Exception as exc:
        return {
            "branch_failures": [_failure("market", subject, exc)],
            "stage_log": [_stage("research_segment")],
        }
    return {"market_findings": {subject: result}, "stage_log": [_stage("research_segment")]}
