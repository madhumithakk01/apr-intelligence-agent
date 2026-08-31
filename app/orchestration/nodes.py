"""Pipeline stage nodes for the orchestration graph.

Most nodes here are still deliberate stubs (IMPLEMENTED_BY records which
future branch replaces each one, and each stub is honest about it rather
than returning a plausible-looking fake result -- a stub that invented
an output would let a later branch's tests pass for the wrong reason).

``ingest`` and ``classify_disclosure`` have been real since
feat/disclosure-classifier (branch 6); ``calibrate_rubrics`` joined them
in feat/rubric-calibration (branch 7); ``score_row`` and
``apply_scoring_kernel`` joined them in feat/qualitative-scoring (branch
8) -- the latter's landed kernel (branch 3) finally had calibrated
qualitative labels to consume instead of a stub's empty output.
``block_capabilities``, ``build_profiles``, ``adjudicate_cluster``, and
``apply_recommendation_policy`` joined them in feat/redundancy-adjudicator
(branch 10) -- the first two were orphaned by feat/redundancy-blocking-
profile (branch 9), which built the modules they call but, based
strictly on refactor/scoring-kernel-consolidation (branch 3), never had
this graph in its own ancestry to wire them into. ``detect_cost_outliers``
and ``explain_cost_outliers`` joined them in feat/cost-outlier-detection
(branch 11). ``build_market_segments`` and ``research_segment`` join them
now, in feat/market-intelligence-agent (branch 12) -- the only genuine
agent in this system (CLAUDE.md section 3): a real LangGraph loop
(app.market_intelligence.graph), not a single call or a bounded ensemble
like every other LLM-backed stage here. LANDED_BY records each
transition to real the same way IMPLEMENTED_BY records a pending one, so
the stage log always says which is which.

Fan-out workers (``classify_disclosure``, ``score_row``,
``adjudicate_cluster``, ``research_segment``) never raise past their own
branch: a failure is recorded in ``branch_failures`` for that subject and
the remaining branches finish normally (CLAUDE.md section 8). Each
delegates its body to a ``_stage_*`` function purely so that isolation
is testable by substituting one. ``calibrate_rubrics``, ``apply_scoring_
kernel``, ``block_capabilities``, ``build_profiles``,
``detect_cost_outliers``, ``explain_cost_outliers``, and
``build_market_segments`` are linear nodes, not fanned out, and have no
such split -- each has exactly one call site (a module-level function
elsewhere) for a test to substitute.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.cost_intelligence import explainability, outlier_detection
from app.disclosure import classifier as disclosure_classifier
from app.ingestion.excel_loader import ExcelLoader
from app.ingestion.row_mapping import build_application_fields
from app.llm.providers import DataSensitivity
from app.market_intelligence import graph as market_intelligence_graph
from app.market_intelligence import segments as market_segments
from app.orchestration import gates
from app.orchestration.state import (
    ClusterTask,
    GraphState,
    RowTask,
    SegmentTask,
    StageStatus,
)
from app.qualitative_scoring import scorer as qualitative_scorer
from app.redundancy import adjudicator, blocking
from app.redundancy import profile_builder as profile_builder_module
from app.redundancy import recommendation_policy
from app.rubric import calibration as rubric_calibration
from app.scoring import kernel

logger = logging.getLogger(__name__)

IMPLEMENTED_BY = {
    "extract_and_ground_products": "feat/product-extraction-grounding (13)",
    "generate_narratives": "feat/narrative-generation (14)",
    "render_report": "feat/report-rendering-consolidation (15)",
}

LANDED_BY = {
    "ingest": "fix/ingestion-integrity (2) + feat/disclosure-classifier (6, wiring)",
    "classify_disclosure": "feat/disclosure-classifier (6)",
    "calibrate_rubrics": "feat/rubric-calibration (7)",
    "score_row": "feat/qualitative-scoring (8)",
    "apply_scoring_kernel": "refactor/scoring-kernel-consolidation (3) + feat/qualitative-scoring (8, wiring)",
    "block_capabilities": "feat/redundancy-blocking-profile (9) + feat/redundancy-adjudicator (10, wiring)",
    "build_profiles": "feat/redundancy-blocking-profile (9) + feat/redundancy-adjudicator (10, wiring)",
    "adjudicate_cluster": "feat/redundancy-adjudicator (10)",
    "apply_recommendation_policy": "feat/redundancy-adjudicator (10)",
    "detect_cost_outliers": "feat/cost-outlier-detection (11)",
    "explain_cost_outliers": "feat/cost-outlier-detection (11)",
    "build_market_segments": "feat/market-intelligence-agent (12)",
    "research_segment": "feat/market-intelligence-agent (12)",
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


def _qualitative_label(field_scores: Dict[str, Any], field: str) -> Optional[str]:
    entry = field_scores.get(field)
    return entry.get("label") if entry else None


def apply_scoring_kernel(state: GraphState) -> Dict[str, Any]:
    """Deterministic TIM-E / COTS-fit scoring (CLAUDE.md section 5).
    app/scoring/kernel.py is the landed implementation (branch 3); this
    is where it is actually called, now that calibrated qualitative
    labels exist to feed it (branch 8). Every qualitative axis comes
    from the scorer's resolved label -- never the row's raw free text --
    so kernel.score_qualitative_label always sees either a label it
    recognizes or None (withheld, or scoring genuinely failed), never a
    value branch 8's own scoring already judged unscorable. Cost fields
    and the security classification come from the disclosure-gated
    application, same as the qualitative scorer's own input, so a
    withheld cost is None here exactly as it is everywhere else."""
    disclosure = state.get("disclosure") or {}
    qualitative_scores = state.get("qualitative_scores") or {}
    results: Dict[str, Any] = {}

    for application in state.get("applications") or []:
        application_id = application.get("application_id")
        if not application_id:
            continue
        gated = (disclosure.get(application_id) or {}).get("gated_application") or application
        field_scores = qualitative_scores.get(application_id) or {}

        inputs = kernel.ScoringInput(
            application_id=application_id,
            application_name=gated.get("application_name") or "",
            business_capability_l2=gated.get("business_capability_l2") or "",
            business_capability_l3=gated.get("business_capability_l3") or "",
            business_criticality=_qualitative_label(field_scores, "business_criticality"),
            strategic_relevance=_qualitative_label(field_scores, "strategic_relevance"),
            business_fitness=_qualitative_label(field_scores, "business_fitness"),
            usage_adoption=_qualitative_label(field_scores, "usage_adoption"),
            application_stability=_qualitative_label(field_scores, "application_stability"),
            maintainability=_qualitative_label(field_scores, "maintainability"),
            availability=_qualitative_label(field_scores, "availability"),
            reliability=_qualitative_label(field_scores, "reliability"),
            scalability=_qualitative_label(field_scores, "scalability"),
            application_security_level=gated.get("application_security_level"),
            skill_availability=_qualitative_label(field_scores, "skill_availability"),
            functional_redundancy=_qualitative_label(field_scores, "functional_redundancy"),
            annual_fte_cost=gated.get("annual_fte_cost"),
            annual_license_cost=gated.get("annual_license_cost"),
            annual_infrastructure_cost=gated.get("annual_infrastructure_cost"),
            other_costs=gated.get("other_costs"),
            # market_product_count: no retrieval exists before branch 12;
            # 0 is the same "no market evidence yet" default the batch
            # path used before this kernel existed (CLAUDE.md section 8).
            market_product_count=0,
        )
        scoring_result = kernel.score_application(inputs)
        results[application_id] = {
            "tim_e_score": scoring_result.tim_e.score,
            "tim_e_decision": scoring_result.tim_e.decision,
            "tim_e_raw_decision": scoring_result.tim_e.raw_decision,
            "floor_applied": scoring_result.tim_e.floor_applied,
            "security_classification": scoring_result.tim_e.security_classification,
            "cots_score": scoring_result.cots.score,
            "cots_recommendation": scoring_result.cots.recommendation,
            "cots_meets_threshold": scoring_result.cots.meets_threshold,
            "modernization_recommendation": scoring_result.modernization_recommendation,
        }

    return {"kernel_results": results, "stage_log": [_stage("apply_scoring_kernel")]}


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
    """Deterministic statistics decide the flag
    (app.cost_intelligence.outlier_detection, branch 11); the minimum
    peer cluster size floor lives in governance_params (section 12).
    Reuses the same capability clusters and normalized cost-per-FTE
    profiles redundancy blocking/profiling already built this run --
    'peer cluster' here is the same concept, not a second one."""
    clusters = state.get("clusters") or []
    profiles = {
        application_id: profile_builder_module.ApplicationProfile.from_dict(data)
        for application_id, data in (state.get("profiles") or {}).items()
    }
    flags = outlier_detection.detect_cost_outliers(clusters, profiles)
    return {
        "cost_outliers": [flag.as_dict() for flag in flags],
        "stage_log": [_stage("detect_cost_outliers")],
    }


def explain_cost_outliers(state: GraphState) -> Dict[str, Any]:
    """Single LLM call per flagged outlier
    (app.cost_intelligence.explainability, branch 11), judging only
    whether the flag is explainable -- it never decides the flag itself.
    Linear, not fanned out (CLAUDE.md section 5's table entry for this
    stage is a single explain call, not an ensemble; the graph's own
    topology never Sends this stage), so a portfolio with several flagged
    outliers makes its calls here sequentially, one node execution.
    Any flag whose explainability comes back needing review
    (governance_params.COST_OUTLIER_EXPLAINABILITY_CONFIDENCE_THRESHOLD)
    enqueues its own gate 4 item."""
    profiles = {
        application_id: profile_builder_module.ApplicationProfile.from_dict(data)
        for application_id, data in (state.get("profiles") or {}).items()
    }
    flags = [
        outlier_detection.CostOutlierFlag(
            application_id=entry["application_id"],
            cluster_id=entry["cluster_id"],
            cost_per_fte=entry["cost_per_fte"],
            direction=entry["direction"],
            cluster_stats=outlier_detection.ClusterCostStats(**entry["cluster_stats"]),
        )
        for entry in (state.get("cost_outliers") or [])
    ]
    explained = explainability.explain_outliers(
        flags, profiles, data_sensitivity=_data_sensitivity(state)
    )

    review_items = [
        {
            "gate": gates.GATE_COST_OUTLIER,
            "subject_id": entry["application_id"],
            "reason": f"{entry['direction']} cost-per-FTE outlier in {entry['cluster_id']}: "
                      f"{entry['explainability']['rationale']}",
            "payload": entry,
        }
        for entry in explained
        if entry["explainability"]["needs_review"]
    ]

    update: Dict[str, Any] = {"cost_outliers": explained, "stage_log": [_stage("explain_cost_outliers")]}
    if review_items:
        update["review_queue"] = review_items
    return update


def build_market_segments(state: GraphState) -> Dict[str, Any]:
    """Deterministic (app.market_intelligence.segments, branch 12): turns
    the redundancy verdicts' typologies into the segments the market
    intelligence agent actually fans out over -- CLAUDE.md section 8's
    "once per redundancy-surviving segment," derived here so the agent
    itself never has to reason about typology cardinality. A new linear
    node ahead of the market fan-out, not folded into
    apply_recommendation_policy: deciding *what to research* is a
    distinct responsibility from the non-compensatory consolidation
    policy that happens to feed it the same verdicts."""
    segments = market_segments.build_segments(
        state.get("verdicts") or [],
        state.get("applications") or [],
        state.get("profiles") or {},
    )
    return {
        "segments": [segment.as_dict() for segment in segments],
        "stage_log": [_stage("build_market_segments")],
    }


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
    """CLAUDE.md section 7. `task["application"]` is already the
    disclosure-gated dict (graph._fan_out_qualitative), never the raw
    one -- app.qualitative_scoring.scorer never sees a withheld value."""
    application = task.get("application") or {}
    application_id = _row_id(task)
    results = qualitative_scorer.score_row(
        application,
        task.get("rubrics"),
        application_id=application_id,
        data_sensitivity=_data_sensitivity(task),
    )
    return {field: result.as_dict() for field, result in results.items()}


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
    """CLAUDE.md sections 3, 5, 8: the only genuine agent in this
    system. Builds and invokes app.market_intelligence.graph's compiled
    subgraph for this one segment, checkpointed on its own thread
    (f"{run_id}:{segment_id}") independently of every other segment's
    -- a provider failure here resumes this branch alone, and the outer
    research_segment wrapper's own try/except (below) is a second,
    outer layer: it catches anything that escapes the subgraph's own
    fail-closed handling (e.g. a genuinely unexpected exception, not
    just a modeled search/assessment failure) so one segment can never
    take the batch down."""
    segment = task.get("segment") or {}
    return market_intelligence_graph.research_segment(
        segment,
        run_id=str(task.get("run_id") or ""),
        data_sensitivity=str(task.get("data_sensitivity") or "real"),
    )


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
    escalating to a 3-sample ensemble on low confidence or rubric
    disagreement (section 7). Any field whose ensemble result carries
    `needs_review` (range >= 2 points) enqueues its own gate 2 review
    item -- one row can enqueue several, one per ambiguous field."""
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
    review_items = [
        {
            "gate": gates.GATE_QUALITATIVE_DISAGREEMENT,
            "subject_id": subject,
            "reason": f"{field}: {field_result.get('rationale', '')}",
            "payload": {"field": field, "field_result": field_result},
        }
        for field, field_result in result.items()
        if field_result.get("needs_review")
    ]
    if review_items:
        update["review_queue"] = review_items
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
