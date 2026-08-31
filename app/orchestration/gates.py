"""The five human-in-the-loop gates -- SPEC.md section 10.

All five are real ``interrupt()`` calls on this branch; what is stubbed
is the evidence that reaches them, because every upstream stage is a
stub (app/orchestration/nodes.py). That split is deliberate: the
mechanism a later branch depends on -- suspend the run, checkpoint it,
resume it with a reviewer decision -- is built and tested now against
synthetic fixtures, while no gate invents a reason to fire.

All review is internal to our own firm. Nothing a gate surfaces routes
back to the client during Phase 1 (SPEC.md section 2).

Gate 1 fires once per engagement and blocks unconditionally until a
reviewer signs the rubrics off. Gates 2-5 are queue-driven: they fire
only when an upstream stage enqueued a ``ReviewItem`` naming them, so a
clean portfolio runs end to end without stopping, and a single ambiguous
row stops the run exactly once, at the right stage.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.types import interrupt

from app.orchestration.state import GraphState, ReviewItem

GATE_RUBRIC_SIGNOFF = "gate_1_rubric_signoff"
GATE_QUALITATIVE_DISAGREEMENT = "gate_2_qualitative_disagreement"
GATE_REDUNDANCY_VERDICT = "gate_3_redundancy_verdict"
GATE_COST_OUTLIER = "gate_4_cost_outlier_explainability"
GATE_NARRATIVE_GROUNDING = "gate_5_narrative_grounding"

ALL_GATES = (
    GATE_RUBRIC_SIGNOFF,
    GATE_QUALITATIVE_DISAGREEMENT,
    GATE_REDUNDANCY_VERDICT,
    GATE_COST_OUTLIER,
    GATE_NARRATIVE_GROUNDING,
)

GATE_REASONS = {
    GATE_RUBRIC_SIGNOFF: "Rubrics must be signed off before any row is scored; frozen for the engagement afterward.",
    GATE_QUALITATIVE_DISAGREEMENT: "Qualitative ensemble range >= 2 points on a 1-5 scale.",
    GATE_REDUNDANCY_VERDICT: (
        "True Duplicate, or a Scale-Tiered Overlap recommending consolidation -- reviewed regardless of "
        "ensemble confidence, because a false merge is far more costly in a client-facing report than an "
        "unnecessary review."
    ),
    GATE_COST_OUTLIER: "Cost-outlier explainability check returned low confidence.",
    GATE_NARRATIVE_GROUNDING: (
        "Narrative failed grounding twice; ships as structured bullets but must be reviewed before the "
        "report is final."
    ),
}


def pending_items(state: GraphState, gate: str) -> List[ReviewItem]:
    return [item for item in (state.get("review_queue") or []) if item.get("gate") == gate]


def _record(gate: str, decision: Any, reviewed: List[ReviewItem]) -> Dict[str, Any]:
    return {
        "gate_decisions": {
            gate: {
                "decision": decision,
                "reviewed_subject_ids": [item.get("subject_id") for item in reviewed],
                "item_count": len(reviewed),
            }
        }
    }


def _queue_driven_gate(state: GraphState, gate: str) -> Dict[str, Any]:
    items = pending_items(state, gate)
    if not items:
        return {}
    decision = interrupt({"gate": gate, "reason": GATE_REASONS[gate], "items": items})
    return _record(gate, decision, items)


def _is_signoff_approved(decision: Any) -> bool:
    """Fail closed (SPEC.md section 2's spirit, applied to a human
    decision): only a recognized approval shape counts as signed off.
    An explicit rejection, or a resume payload this gate simply doesn't
    recognize, is treated as NOT approved -- never assumed. This does
    not change what gets recorded in gate_decisions, which is always the
    decision verbatim; it only decides what `rubrics["status"]` and
    `rubric_signoff["signed_off"]` become."""
    if decision == "approved":
        return True
    if isinstance(decision, dict):
        if decision.get("action") in {"approve", "approved"}:
            return True
        if decision.get("signed_off") is True:
            return True
    return False


def gate_rubric_signoff(state: GraphState) -> Dict[str, Any]:
    """Gate 1. Unconditional, once per engagement. Already-signed-off
    rubrics are frozen, so a resumed or re-run engagement does not stop
    here a second time.

    Approving freezes the proposed rubric to "signed_off" -- the state
    app.qualitative_scoring (branch 8) must require before trusting it.
    Rejecting freezes it to "rejected" instead: the proposal stays in
    state for a reviewer to see what was rejected and why, but nothing
    downstream may score against it."""
    if (state.get("gate_decisions") or {}).get(GATE_RUBRIC_SIGNOFF):
        return {}
    proposed = state.get("rubrics") or {}
    decision = interrupt(
        {
            "gate": GATE_RUBRIC_SIGNOFF,
            "reason": GATE_REASONS[GATE_RUBRIC_SIGNOFF],
            "rubrics": proposed,
        }
    )
    approved = _is_signoff_approved(decision)
    frozen_rubrics = dict(proposed)
    frozen_rubrics["status"] = "signed_off" if approved else "rejected"
    return {
        "gate_decisions": {GATE_RUBRIC_SIGNOFF: {"decision": decision, "item_count": 0}},
        "rubric_signoff": {"signed_off": approved, "decision": decision},
        "rubrics": frozen_rubrics,
    }


def gate_qualitative_disagreement(state: GraphState) -> Dict[str, Any]:
    """Gate 2. Fires on the rows whose ensemble range was >= 2 points
    (SPEC.md section 7) -- one stop for the whole batch of them, not
    one stop per row, since the fan-out has already joined here."""
    return _queue_driven_gate(state, GATE_QUALITATIVE_DISAGREEMENT)


def gate_redundancy_verdict(state: GraphState) -> Dict[str, Any]:
    """Gate 3."""
    return _queue_driven_gate(state, GATE_REDUNDANCY_VERDICT)


def gate_cost_outlier(state: GraphState) -> Dict[str, Any]:
    """Gate 4."""
    return _queue_driven_gate(state, GATE_COST_OUTLIER)


def gate_narrative_grounding(state: GraphState) -> Dict[str, Any]:
    """Gate 5. Unlike gates 1-3 this one does not withhold an output --
    the narrative already fell back to structured bullets -- but the
    report is not final until a reviewer has seen it."""
    return _queue_driven_gate(state, GATE_NARRATIVE_GROUNDING)
