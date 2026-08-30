"""Pass-through stage stubs for the orchestration graph.

Every node here is a deliberate stub. This branch builds the control
flow -- topology, checkpointing, fan-out, and the five interrupt()
gates -- and nothing else: no LLM call, no scoring, no market lookup.
The stubs exist so the state machine can be executed and tested against
synthetic fixtures before a single real stage is wired into it, which is
also why they are honest about it: each records
``implemented_by_branch`` in the stage log rather than returning a
plausible-looking fake result. A stub that invented an output would let
a later branch's tests pass for the wrong reason.

IMPLEMENTED_BY below maps each stage to the branch that replaces its
stub (numbering per the repository's branch sequence).

Fan-out workers (``classify_disclosure``, ``score_row``,
``adjudicate_cluster``, ``research_segment``) never raise past their own
branch: a failure is recorded in ``branch_failures`` for that subject and
the remaining branches finish normally (CLAUDE.md section 8). Each
delegates its body to a ``_stage_*`` function purely so that isolation
is testable by substituting one.
"""

from __future__ import annotations

from typing import Any, Dict

from app.orchestration.state import (
    ClusterTask,
    GraphState,
    RowTask,
    SegmentTask,
    StageStatus,
)

IMPLEMENTED_BY = {
    "ingest": "fix/ingestion-integrity (2, landed -- wiring deferred to branch 6)",
    "classify_disclosure": "feat/disclosure-classifier (6)",
    "calibrate_rubrics": "feat/rubric-calibration (7)",
    "score_row": "feat/qualitative-scoring (8)",
    "apply_scoring_kernel": "refactor/scoring-kernel-consolidation (3, landed -- wiring deferred to branch 8)",
    "block_capabilities": "feat/redundancy-blocking-profile (9)",
    "build_profiles": "feat/redundancy-blocking-profile (9)",
    "adjudicate_cluster": "feat/redundancy-adjudicator (10)",
    "apply_recommendation_policy": "feat/redundancy-adjudicator (10)",
    "detect_cost_outliers": "feat/cost-outlier-detection (11)",
    "explain_cost_outliers": "feat/cost-outlier-detection (11)",
    "research_segment": "feat/market-intelligence-agent (12)",
    "extract_and_ground_products": "feat/product-extraction-grounding (13)",
    "generate_narratives": "feat/narrative-generation (14)",
    "render_report": "feat/report-rendering-consolidation (15)",
}


def _stage(name: str) -> StageStatus:
    return StageStatus(stage=name, status="stub", implemented_by_branch=IMPLEMENTED_BY[name])


def _failure(kind: str, subject_id: str, error: BaseException) -> Dict[str, Any]:
    return {"branch_kind": kind, "subject_id": subject_id, "error": f"{type(error).__name__}: {error}"}


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
    """Deterministic, generous capability blocking. Coarse on purpose --
    a pairing missed here can never be recovered downstream (section 5)."""
    return {"clusters": list(state.get("clusters") or []), "stage_log": [_stage("block_capabilities")]}


def build_profiles(state: GraphState) -> Dict[str, Any]:
    """Multi-axis profile building -- five axes kept separate, never
    pre-blended into a single similarity score (section 9)."""
    return {"stage_log": [_stage("build_profiles")]}


def apply_recommendation_policy(state: GraphState) -> Dict[str, Any]:
    """Deterministic, non-compensatory, ordered policy (section 9):
    classification gate, then criticality ceiling, then normalized cost,
    then technical feasibility."""
    return {
        "recommendations": list(state.get("recommendations") or []),
        "stage_log": [_stage("apply_recommendation_policy")],
    }


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
    return {}


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
    """One fanned-out branch per candidate cluster. 3-sample ensemble
    into the five-way typology (section 9) -- an ensemble, not an agent."""
    subject = str((task.get("cluster") or {}).get("cluster_id") or "unknown")
    try:
        result = _stage_adjudication(task)
    except Exception as exc:
        return {
            "branch_failures": [_failure("redundancy", subject, exc)],
            "stage_log": [_stage("adjudicate_cluster")],
        }
    update: Dict[str, Any] = {"stage_log": [_stage("adjudicate_cluster")]}
    verdict = result.get("verdict")
    if verdict:
        update["verdicts"] = [verdict]
    review = result.get("review_items")
    if review:
        update["review_queue"] = list(review)
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
