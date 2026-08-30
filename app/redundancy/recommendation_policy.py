"""Non-compensatory recommendation policy -- CLAUDE.md sections 9, 10.

Deterministic, applied after the typology label; ordered, not weighted,
so savings can never mask a compliance problem (section 9):

  1. Data classification gate -- a classification mismatch blocks
     consolidation outright, full stop.
  2. Criticality ceiling -- never recommend moving a high-criticality
     workload onto a platform with unproven stability at that load
     purely to save license cost.
  3. Normalized cost-vs-scale -- cost enters only here, as
     cost-per-unit-of-scale, post-normalization.
  4. Technical feasibility -- tech stack/maintainability decide *how*
     consolidation happens, not *whether* it should (never blocking).

These gates only have a decision to make for the two typologies where
consolidation is even on the table (True Duplicate, Scale-Tiered
Overlap). Partial/Component Overlap, Distinct, Indeterminate, and
Adjudication Failed each have one fixed recommendation directly off the
typology -- CLAUDE.md section 9's own typology table already states it.

A gate that cannot read what it needs (an unrecognized qualitative label
-- this branch has no access to feat/qualitative-scoring's calibrated
rubrics, only app.scoring.kernel's five fixed canonical labels) treats
that the same as the deterministic pre-check in adjudicator.py treats a
missing value: refuse to authorize consolidation and require review,
never guess which way an unreadable label would have gone.

evaluate() is what finalizes gate 3 (CLAUDE.md section 10): a verdict's
own `mandatory_review` already covers Indeterminate, Adjudication
Failed, full disagreement, and a True-Duplicate majority
(adjudicator.py); this module adds the other named trigger --
Scale-Tiered Overlap actually recommending consolidation (migrating the
light tier), "regardless of ensemble confidence."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from app.redundancy.adjudicator import (
    ADJUDICATION_FAILED,
    DISTINCT,
    INDETERMINATE_WITHHELD_DATA,
    PARTIAL_COMPONENT_OVERLAP,
    SCALE_TIERED_OVERLAP,
    TRUE_DUPLICATE,
    AdjudicationVerdict,
)
from app.redundancy.profile_builder import ApplicationProfile
from app.scoring.kernel import score_qualitative_label

_HIGH_CRITICALITY_POINTS = {4, 5}  # "high", "very high"
_PROVEN_STABILITY_POINTS = {4, 5}  # "high", "very high" -- anything below is "unproven" for this gate


@dataclass(frozen=True)
class RecommendationResult:
    typology: str
    recommendation: str
    consolidation_blocked_by: Optional[str]
    """Which gate blocked consolidation, if any:
    "classification_mismatch" | "criticality_ceiling" |
    "cost_does_not_justify" | None (nothing blocked it, or there was
    never anything to block -- see recommendation for which)."""
    mandatory_review: bool
    """The final CLAUDE.md section 10 gate-3 trigger -- the verdict's
    own mandatory_review, plus Scale-Tiered Overlap actually recommending
    consolidation."""
    phase2_discovery: bool
    """Whether this pair belongs on the Phase 2 discovery log (CLAUDE.md
    section 9): Partial/Component Overlap always does; Indeterminate and
    Adjudication Failed always do (there is unresolved work here);
    Distinct never does; True Duplicate / Scale-Tiered Overlap do only
    when a gate blocked full consolidation (there's a follow-up
    question, not just a closed no-action case)."""
    rationale: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "typology": self.typology,
            "recommendation": self.recommendation,
            "consolidation_blocked_by": self.consolidation_blocked_by,
            "mandatory_review": self.mandatory_review,
            "phase2_discovery": self.phase2_discovery,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class _GateOutcome:
    blocked: bool
    reason: Optional[str]
    recommendation: Optional[str]


def _heavier_lighter(
    a: ApplicationProfile, b: ApplicationProfile
) -> Tuple[ApplicationProfile, ApplicationProfile]:
    """(heavier/target-platform, lighter/migration-candidate) by FTE
    Count, falling back to cost-per-FTE, falling back to input order.
    For True Duplicate this designation is an arbitrary but deterministic
    tie-break -- the typology's own trigger is "comparable scale," so
    there is no real heavy/light distinction to find; the gates below
    still need *some* consistent direction to reason in."""
    a_fte = a.scale_usage.fte_count
    b_fte = b.scale_usage.fte_count
    if a_fte is not None and b_fte is not None and a_fte != b_fte:
        return (a, b) if a_fte > b_fte else (b, a)
    a_cost = a.cost.cost_per_fte
    b_cost = b.cost.cost_per_fte
    if a_cost is not None and b_cost is not None and a_cost != b_cost:
        return (a, b) if a_cost > b_cost else (b, a)
    return (a, b)


def _classification_gate(a: ApplicationProfile, b: ApplicationProfile) -> _GateOutcome:
    class_a = a.risk_classification.application_security_level
    class_b = b.risk_classification.application_security_level
    if class_a is None or class_b is None:
        # Defensive: adjudicator.py's Indeterminate pre-check already
        # screens for this before an ensemble ever runs, so evaluate()
        # should never reach here with either unknown -- this branch
        # exists only so a direct caller of this module gets the same
        # fail-safe rather than a crash.
        return _GateOutcome(True, "classification_unknown", "Retain both -- data classification is not confirmed for both applications.")
    if class_a.strip().casefold() != class_b.strip().casefold():
        return _GateOutcome(True, "classification_mismatch", "Retain both -- data classification mismatch precludes consolidation.")
    return _GateOutcome(False, None, None)


def _criticality_ceiling(a: ApplicationProfile, b: ApplicationProfile) -> _GateOutcome:
    heavier, lighter = _heavier_lighter(a, b)
    lighter_criticality = score_qualitative_label(lighter.scale_usage.business_criticality)
    heavier_stability = score_qualitative_label(heavier.risk_classification.application_stability)
    if lighter_criticality is None or heavier_stability is None:
        return _GateOutcome(
            True,
            "criticality_or_stability_unreadable",
            "Retain both -- criticality or stability could not be confidently read from the available "
            "labels; do not consolidate without review.",
        )
    if lighter_criticality in _HIGH_CRITICALITY_POINTS and heavier_stability not in _PROVEN_STABILITY_POINTS:
        return _GateOutcome(
            True,
            "criticality_ceiling",
            "Retain both -- the lighter-scale application's criticality exceeds what the heavier "
            "platform's proven stability supports.",
        )
    return _GateOutcome(False, None, None)


def _cost_justifies_consolidation(a: ApplicationProfile, b: ApplicationProfile) -> _GateOutcome:
    heavier, lighter = _heavier_lighter(a, b)
    heavier_cost = heavier.cost.cost_per_fte
    lighter_cost = lighter.cost.cost_per_fte
    if heavier_cost is None or lighter_cost is None:
        # Defensive, same reasoning as _classification_gate: the
        # Indeterminate pre-check already screens for unknown cost.
        return _GateOutcome(True, "cost_unknown", "Retain both -- normalized cost is not confirmed for both applications.")
    if lighter_cost <= heavier_cost:
        return _GateOutcome(
            True,
            "cost_does_not_justify",
            "Retain both -- the lighter tier's normalized cost is not higher than the heavier "
            "platform's, so consolidation would not clearly reduce cost.",
        )
    return _GateOutcome(False, None, None)


def _technical_feasibility_note(a: ApplicationProfile, b: ApplicationProfile) -> str:
    """Never blocks (CLAUDE.md section 9: decides *how*, not *whether*)
    -- always returns guidance text to append to the recommendation."""
    shared_stack = set(a.technical.technology_stack_tokens) & set(b.technical.technology_stack_tokens)
    if shared_stack:
        return "Shared technology stack components suggest a comparatively straightforward migration path."
    return "No shared technology stack components were detected -- treat migration feasibility as unproven and scope it explicitly."


_FIXED_RECOMMENDATIONS = {
    PARTIAL_COMPONENT_OVERLAP: "Retain both; log for Phase 2 scoping.",
    DISTINCT: "No action.",
    INDETERMINATE_WITHHELD_DATA: "No verdict -- withheld data; mandatory human review; logged as a Phase 2 discovery item.",
    ADJUDICATION_FAILED: "No verdict -- adjudication did not complete; re-run and route to review.",
}

_PHASE2_BY_FIXED_TYPOLOGY = {
    PARTIAL_COMPONENT_OVERLAP: True,
    DISTINCT: False,
    INDETERMINATE_WITHHELD_DATA: True,
    ADJUDICATION_FAILED: True,
}


def evaluate(
    verdict: AdjudicationVerdict, profile_a: ApplicationProfile, profile_b: ApplicationProfile
) -> RecommendationResult:
    typology = verdict.typology

    if typology in _FIXED_RECOMMENDATIONS:
        return RecommendationResult(
            typology=typology,
            recommendation=_FIXED_RECOMMENDATIONS[typology],
            consolidation_blocked_by=None,
            mandatory_review=verdict.mandatory_review,
            phase2_discovery=_PHASE2_BY_FIXED_TYPOLOGY[typology],
            rationale=verdict.rationale,
        )

    # True Duplicate or Scale-Tiered Overlap: the ordered, non-compensatory gates.
    for gate in (_classification_gate, _criticality_ceiling, _cost_justifies_consolidation):
        outcome = gate(profile_a, profile_b)
        if outcome.blocked:
            recommendation = outcome.recommendation
            if typology == SCALE_TIERED_OVERLAP:
                recommendation = recommendation.replace("Retain both", "Retain both as differentiated tiers")
            return RecommendationResult(
                typology=typology,
                recommendation=recommendation,
                consolidation_blocked_by=outcome.reason,
                mandatory_review=verdict.mandatory_review,
                phase2_discovery=True,
                rationale=outcome.recommendation,
            )

    feasibility_note = _technical_feasibility_note(profile_a, profile_b)
    if typology == TRUE_DUPLICATE:
        recommendation = f"Consolidate; retire one application. {feasibility_note}"
    else:  # SCALE_TIERED_OVERLAP, nothing blocked consolidation
        recommendation = f"Migrate the lighter tier onto the heavier platform. {feasibility_note}"

    return RecommendationResult(
        typology=typology,
        recommendation=recommendation,
        consolidation_blocked_by=None,
        mandatory_review=True,  # CLAUDE.md section 10: always, regardless of ensemble confidence
        phase2_discovery=False,
        rationale="No gate blocked consolidation.",
    )
