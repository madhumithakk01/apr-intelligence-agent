"""The five human-in-the-loop gates -- CLAUDE.md section 10.

All five are real ``interrupt()`` calls on this branch; what is stubbed
is the evidence that reaches them, because every upstream stage is a
stub (app/orchestration/nodes.py). That split is deliberate: the
mechanism a later branch depends on -- suspend the run, checkpoint it,
resume it with a reviewer decision -- is built and tested now against
synthetic fixtures, while no gate invents a reason to fire.

All review is internal to our own firm. Nothing a gate surfaces routes
back to the client during Phase 1 (CLAUDE.md section 2).

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


def gate_rubric_signoff(state: GraphState) -> Dict[str, Any]:
    """Gate 1. Unconditional, once per engagement. Already-signed-off
    rubrics are frozen, so a resumed or re-run engagement does not stop
    here a second time."""
    if (state.get("gate_decisions") or {}).get(GATE_RUBRIC_SIGNOFF):
        return {}
    decision = interrupt(
        {
            "gate": GATE_RUBRIC_SIGNOFF,
            "reason": GATE_REASONS[GATE_RUBRIC_SIGNOFF],
            "rubrics": state.get("rubrics") or {},
        }
    )
    return {
        "gate_decisions": {GATE_RUBRIC_SIGNOFF: {"decision": decision, "item_count": 0}},
        "rubric_signoff": {"signed_off": True, "decision": decision},
    }


def gate_qualitative_disagreement(state: GraphState) -> Dict[str, Any]:
    """Gate 2. Fires on the rows whose ensemble range was >= 2 points
    (CLAUDE.md section 7) -- one stop for the whole batch of them, not
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
