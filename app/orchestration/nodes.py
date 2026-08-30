"""Pipeline stage nodes for the orchestration graph.

Most nodes here are still deliberate stubs (IMPLEMENTED_BY records which
future branch replaces each one, and each stub is honest about it rather
than returning a plausible-looking fake result -- a stub that invented
an output would let a later branch's tests pass for the wrong reason).
``ingest`` and ``classify_disclosure`` have been real since
feat/disclosure-classifier (branch 6); ``calibrate_rubrics`` joins them
as of feat/rubric-calibration (branch 7). LANDED_BY records that
transition the same way IMPLEMENTED_BY records a pending one, so the
stage log always says which is which.

Fan-out workers (``classify_disclosure``, ``score_row``,
``adjudicate_cluster``, ``research_segment``) never raise past their own
branch: a failure is recorded in ``branch_failures`` for that subject and
the remaining branches finish normally (CLAUDE.md section 8). Each
delegates its body to a ``_stage_*`` function purely so that isolation
is testable by substituting one. ``calibrate_rubrics`` is a linear node,
not fanned out, and has no such split -- there is exactly one call site
(app.rubric.calibration.calibrate_rubrics) for a test to substitute.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.disclosure import classifier as disclosure_classifier
from app.ingestion.excel_loader import ExcelLoader
from app.ingestion.row_mapping import build_application_fields
from app.llm.providers import DataSensitivity
from app.orchestration.state import (
    ClusterTask,
    GraphState,
    RowTask,
    SegmentTask,
    StageStatus,
)
from app.rubric import calibration as rubric_calibration

logger = logging.getLogger(__name__)

IMPLEMENTED_BY = {
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

LANDED_BY = {
    "ingest": "fix/ingestion-integrity (2) + feat/disclosure-classifier (6, wiring)",
    "classify_disclosure": "feat/disclosure-classifier (6)",
    "calibrate_rubrics": "feat/rubric-calibration (7)",
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
    """Applications supplied directly in the run's input state pass
    through unchanged -- this is how tests and any future caller that
    already holds rows (e.g. branch 16's async submit endpoint, once it
    exists) skip the file entirely. Otherwise, load `dataset_path`
    through the real deterministic loader (CLAUDE.md section 5):
    ExcelLoader, duplicate-Application-ID surfacing, and safe cost/count
    parsing (app.ingestion.row_mapping) that treats a non-numeric cost
    cell as unparsed rather than crashing the batch (section 4 bug 6).

    Colliding Application IDs are excluded from `applications` and
    recorded in `ingestion_collisions` -- never silently dropped in favor
    of a first-write winner (section 4 bug 7)."""
    applications = state.get("applications") or []
    if applications:
        return {"applications": list(applications), "stage_log": [_stage("ingest")]}

    dataset_path = state.get("dataset_path")
    if not dataset_path:
        return {"applications": [], "stage_log": [_stage("ingest")]}

    frame = ExcelLoader(dataset_path).load()
    collisions = ExcelLoader.find_duplicate_application_ids(frame)
    skip_ids = set(collisions.keys())

    loaded: list = []
    for _, row in frame.iterrows():
        application_id = str(row["Application ID"]).strip()
        if application_id in skip_ids:
            continue
        loaded.append(build_application_fields(row, application_id=application_id))

    collision_records = [
        {"application_id": application_id, "occurrences": occurrences}
        for application_id, occurrences in collisions.items()
    ]
    return {
        "applications": loaded,
        "ingestion_collisions": collision_records,
        "stage_log": [_stage("ingest")],
    }


def calibrate_rubrics(state: GraphState) -> Dict[str, Any]:
    """CLAUDE.md section 5/7: single structured LLM call, once per field
    per engagement, over disclosure-gated applications (section 6) so a
    withheld/unknown/placeholder value can never become a rubric anchor.
    Proposed here; frozen to "signed_off" or "rejected" by gate 1
    (gates.gate_rubric_signoff) before any row is scored.

    A row whose disclosure branch failed (absent from state["disclosure"]
    entirely) contributes nothing to calibration -- there is no gated
    value to trust for it, so it is simply skipped rather than falling
    back to its raw, ungated value."""
    gated_applications = [
        entry["gated_application"]
        for entry in (state.get("disclosure") or {}).values()
        if "gated_application" in entry
    ]
    rubrics = rubric_calibration.calibrate_rubrics(
        gated_applications, data_sensitivity=_data_sensitivity(state)
    )
    return {
        "rubrics": {
            "status": "proposed",
            "fields": {field: field_rubric.as_dict() for field, field_rubric in rubrics.items()},
        },
        "stage_log": [_stage("calibrate_rubrics")],
    }


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
    """CLAUDE.md section 6. Single structured LLM call per row
    (app/disclosure/classifier.py), gating every field it classifies out
    of downstream scoring unless the client actually answered it."""
    application = task.get("application") or {}
    application_id = _row_id(task)
    results = disclosure_classifier.classify_row(
        application, application_id=application_id, data_sensitivity=_data_sensitivity(task)
    )
    return {
        "results": {field: result.as_dict() for field, result in results.items()},
        "gated_application": disclosure_classifier.apply_disclosure_gate(application, results),
        "phase2_agenda": disclosure_classifier.build_phase2_agenda(
            application_id, application.get("application_name"), results
        ),
    }


def _stage_qualitative(task: RowTask) -> Dict[str, Any]:
    return {}


def _stage_adjudication(task: ClusterTask) -> Dict[str, Any]:
    return {}


def _stage_market_research(task: SegmentTask) -> Dict[str, Any]:
    return {}


def _row_id(task: RowTask) -> str:
    return str((task.get("application") or {}).get("application_id") or "unknown")


def classify_disclosure(task: RowTask) -> Dict[str, Any]:
    """One fanned-out branch per application row (section 6). A
    classification failure for this row is recorded and the row's
    disclosure result is simply absent from state -- every field on it
    stays unscored downstream (never defaulted to Answered), and the
    other rows' branches are unaffected (section 8)."""
    subject = _row_id(task)
    try:
        result = _stage_disclosure(task)
    except Exception as exc:  # branch-local: never fails the batch
        return {
            "branch_failures": [_failure("disclosure", subject, exc)],
            "stage_log": [_stage("classify_disclosure")],
        }
    update: Dict[str, Any] = {"disclosure": {subject: result}, "stage_log": [_stage("classify_disclosure")]}
    agenda = result.get("phase2_agenda")
    if agenda:
        update["phase2_agenda"] = list(agenda)
    return update


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
