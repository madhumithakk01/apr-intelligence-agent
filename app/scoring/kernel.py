"""Scoring kernel -- SPEC.md section 13.

Single source of truth for TIM-E scoring, COTS-fit scoring, and the
skill-availability floor. Consolidates the two divergent engines that
used to live in app/services/agent_service.py (API path) and
app/services/analysis_service.py (batch path) -- SPEC.md section 4,
bug 2.

Deterministic only, no LLM calls. Real qualitative LLM scoring
(SPEC.md section 7: single call, 3-sample ensemble escalation on low
confidence) is feat/qualitative-scoring (branch 8) --
score_qualitative_label below is this branch's explicit, honestly
labeled deferral point for that future replacement, matching how
branch 1 deferred LangSmith tracing and branch 2 deferred the full
Disclosure Classifier. It is not a stand-in implementation of branch 8.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.scoring import governance_params as gp

QUALITATIVE_LABELS = {
    "very high": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "very low": 1,
}
"""The only 5 labels this deterministic kernel can score. Anything else
-- including SPEC.md's own cited real examples ("too risky," "Somewhat
cumbersome," "cannot say") -- is unscored (None), never defaulted
(SPEC.md section 4, bug 1)."""


def score_qualitative_label(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return QUALITATIVE_LABELS.get(cleaned)


@dataclass(frozen=True)
class ScoringInput:
    application_id: str
    application_name: str
    business_capability_l2: str
    business_capability_l3: str

    business_criticality: Optional[str]
    strategic_relevance: Optional[str]
    business_fitness: Optional[str]
    usage_adoption: Optional[str]

    application_stability: Optional[str]
    maintainability: Optional[str]
    availability: Optional[str]
    reliability: Optional[str]
    scalability: Optional[str]

    application_security_level: Optional[str]
    skill_availability: Optional[str]
    functional_redundancy: Optional[str]

    annual_fte_cost: Optional[float]
    annual_license_cost: Optional[float]
    annual_infrastructure_cost: Optional[float]
    other_costs: Optional[float]

    market_product_count: int = 0
    """Count of products already retrieved by the caller's own market
    intelligence integration (agent_service.py's MarketService/Tavily
    call, out of this branch's scope -- SPEC.md section 8). The
    kernel never calls MarketService and never sees product objects,
    only this count. Callers that don't perform retrieval (the batch
    path) pass 0."""


@dataclass(frozen=True)
class AxisScore:
    name: str
    value: Optional[int]


@dataclass(frozen=True)
class AggregateScore:
    value: Optional[float]
    scored_axes: List[str] = field(default_factory=list)
    unscored_axes: List[str] = field(default_factory=list)
    status: str = "insufficient_data"  # "complete" | "partial" | "insufficient_data"


def _average_axes(axes: List[AxisScore]) -> AggregateScore:
    """Average only the scored axes; refuse to produce a number below a
    strict majority scored (ceil(n/2)) -- a structural cutoff, not an
    empirically-tuned one, so it needs no golden-subset validation."""
    scored = [a for a in axes if a.value is not None]
    scored_names = [a.name for a in scored]
    unscored_names = [a.name for a in axes if a.value is None]

    if len(scored) < math.ceil(len(axes) / 2):
        return AggregateScore(None, scored_names, unscored_names, "insufficient_data")

    average = sum(a.value for a in scored) / len(scored)
    status = "complete" if not unscored_names else "partial"
    return AggregateScore(average, scored_names, unscored_names, status)


@dataclass(frozen=True)
class CostAggregate:
    total_known: Optional[float]
    missing_fields: List[str]
    is_complete: bool


def _aggregate_cost(
    fte: Optional[float],
    license_: Optional[float],
    infra: Optional[float],
    other: Optional[float],
) -> CostAggregate:
    fields = {
        "annual_fte_cost": fte,
        "annual_license_cost": license_,
        "annual_infrastructure_cost": infra,
        "other_costs": other,
    }
    known = {name: value for name, value in fields.items() if value is not None}
    missing = [name for name in fields if name not in known]
    if not known:
        return CostAggregate(None, missing, False)
    return CostAggregate(sum(known.values()), missing, not missing)


def _cost_bucket(cost: CostAggregate) -> Optional[int]:
    """None if every cost component is withheld/unparsed -- never
    guessed as bucket 1, which would silently understate a withheld
    high cost (SPEC.md section 2)."""
    if cost.total_known is None:
        return None
    total = cost.total_known
    if total >= 800_000:
        return 5
    if total >= 500_000:
        return 4
    if total >= 250_000:
        return 3
    if total >= 100_000:
        return 2
    return 1


@dataclass(frozen=True)
class TimEResult:
    score: Optional[float]
    raw_decision: Optional[str]
    decision: str
    value_score: AggregateScore
    health_score: AggregateScore
    consolidation_need: AggregateScore
    floor_applied: Optional[str]
    security_classification: Optional[str]


def _decision_from_score(score: float) -> str:
    if score >= gp.DECISION_THRESHOLDS["invest"]:
        return "Invest"
    if score >= gp.DECISION_THRESHOLDS["migrate"]:
        return "Migrate"
    if score >= gp.DECISION_THRESHOLDS["tolerate"]:
        return "Tolerate"
    return "Eliminate"


def _apply_skill_availability_floor(
    raw_decision: str,
    skill_availability: Optional[str],
    application_stability: Optional[str],
) -> Tuple[str, Optional[str]]:
    """SPEC.md section 4 bug 5: low skill availability + fragile
    stability forces a minimum "Migrate," overriding a high raw score.
    Checked against the raw stability axis, not the blended
    health_score, so folding Availability/Reliability/Scalability in
    can never dilute a genuinely fragile stability signal out of this
    gate. Both inputs must actually be scored -- absence of evidence
    never triggers the floor."""
    skill = score_qualitative_label(skill_availability)
    stability = score_qualitative_label(application_stability)
    if skill is None or stability is None:
        return raw_decision, None
    if skill <= 2 and stability <= 2 and raw_decision == "Invest":
        return "Migrate", "skill_availability_floor"
    return raw_decision, None


def compute_tim_e(inputs: ScoringInput) -> TimEResult:
    value_score = _average_axes(
        [
            AxisScore("business_criticality", score_qualitative_label(inputs.business_criticality)),
            AxisScore("strategic_relevance", score_qualitative_label(inputs.strategic_relevance)),
            AxisScore("business_fitness", score_qualitative_label(inputs.business_fitness)),
            AxisScore("usage_adoption", score_qualitative_label(inputs.usage_adoption)),
        ]
    )
    health_score = _average_axes(
        [
            AxisScore("application_stability", score_qualitative_label(inputs.application_stability)),
            AxisScore("maintainability", score_qualitative_label(inputs.maintainability)),
            AxisScore("availability", score_qualitative_label(inputs.availability)),
            AxisScore("reliability", score_qualitative_label(inputs.reliability)),
            AxisScore("scalability", score_qualitative_label(inputs.scalability)),
            # application_security_level intentionally excluded -- SPEC.md
            # section 4 bug 4. It is a data-classification label, not a
            # technical-health axis; see security_classification below.
        ]
    )
    cost = _aggregate_cost(
        inputs.annual_fte_cost,
        inputs.annual_license_cost,
        inputs.annual_infrastructure_cost,
        inputs.other_costs,
    )
    consolidation_need = _average_axes(
        [
            AxisScore("functional_redundancy", score_qualitative_label(inputs.functional_redundancy)),
            AxisScore("cost_bucket", _cost_bucket(cost)),
        ]
    )

    if value_score.value is None or health_score.value is None or consolidation_need.value is None:
        return TimEResult(
            score=None,
            raw_decision=None,
            decision="Insufficient Data",
            value_score=value_score,
            health_score=health_score,
            consolidation_need=consolidation_need,
            floor_applied=None,
            security_classification=inputs.application_security_level,
        )

    raw_score = round(
        (
            value_score.value * gp.TIME_WEIGHTS["value"]
            + health_score.value * gp.TIME_WEIGHTS["health"]
            + (6 - consolidation_need.value) * gp.TIME_WEIGHTS["consolidation"]
        )
        * 20,
        2,
    )
    raw_decision = _decision_from_score(raw_score)
    decision, floor_applied = _apply_skill_availability_floor(
        raw_decision, inputs.skill_availability, inputs.application_stability
    )
    return TimEResult(
        score=raw_score,
        raw_decision=raw_decision,
        decision=decision,
        value_score=value_score,
        health_score=health_score,
        consolidation_need=consolidation_need,
        floor_applied=floor_applied,
        security_classification=inputs.application_security_level,
    )


@dataclass(frozen=True)
class CotsFitResult:
    score: Optional[float]
    recommendation: str
    meets_threshold: bool
    unscored_axes: List[str] = field(default_factory=list)


def compute_cots_fit(inputs: ScoringInput) -> CotsFitResult:
    if inputs.market_product_count <= 0:
        return CotsFitResult(
            score=None,
            recommendation="Retain existing application — insufficient retrieved market data for COTS comparison",
            meets_threshold=False,
        )

    axes = {
        "functional_redundancy": score_qualitative_label(inputs.functional_redundancy),
        "maintainability": score_qualitative_label(inputs.maintainability),
        "application_stability": score_qualitative_label(inputs.application_stability),
    }
    unscored = [name for name, value in axes.items() if value is None]
    if unscored:
        return CotsFitResult(
            score=None,
            recommendation="Insufficient qualitative data for COTS-fit scoring",
            meets_threshold=False,
            unscored_axes=unscored,
        )

    weights = gp.COTS_FIT_WEIGHTS
    base_score = (
        (6 - axes["functional_redundancy"]) * weights["functional_redundancy"]
        + (6 - axes["maintainability"]) * weights["maintainability"]
        + (6 - axes["application_stability"]) * weights["application_stability"]
    ) * 20
    bonus = min(inputs.market_product_count, gp.MARKET_PRODUCT_BONUS_CAP) * gp.MARKET_PRODUCT_BONUS_PER_PRODUCT
    score = round(min(base_score + bonus, 100), 2)
    meets_threshold = score >= gp.COTS_REPLACE_THRESHOLD
    recommendation = "Replace with COTS" if meets_threshold else "Retain/Enhance Existing Application"
    return CotsFitResult(score=score, recommendation=recommendation, meets_threshold=meets_threshold)


def recommend_modernization_path(tim_e: TimEResult, cots: CotsFitResult, market_product_count: int) -> str:
    """Single canonical modernization category, replacing
    agent_service.py's old _modernization and analysis_service.py's old
    _modernization_choice (SPEC.md section 4 bugs 2 and 3 -- the
    latter's separate `cots_score >= 70` check was the exact source of
    the 65-vs-70 contradiction; compute_cots_fit above is now the only
    place in the codebase that compares a COTS score to a threshold)."""
    if tim_e.decision == "Insufficient Data":
        return "Insufficient qualitative data to recommend a modernization path — see unscored axes."
    if tim_e.decision == "Eliminate":
        return "Retire over a phased transition plan."
    if cots.meets_threshold:
        return "Evaluate the leading retrieved COTS candidate via fit-gap workshop, POC, and phased data migration."
    if market_product_count == 0:
        return "Retain and monitor; complete targeted COTS research before a replacement decision."
    if tim_e.health_score.value is not None and tim_e.health_score.value < 2.5:
        return "Re-architect/refactor core components for resilience."
    if tim_e.decision == "Invest":
        return "Invest in refactoring, API hardening, and observability."
    if tim_e.decision == "Migrate":
        return "Migrate to cloud-native architecture with minimal downtime."
    return "Tolerate short-term while preparing modernization backlog."


@dataclass(frozen=True)
class ScoringResult:
    tim_e: TimEResult
    cots: CotsFitResult
    modernization_recommendation: str


def score_application(inputs: ScoringInput) -> ScoringResult:
    tim_e = compute_tim_e(inputs)
    cots = compute_cots_fit(inputs)
    modernization = recommend_modernization_path(tim_e, cots, inputs.market_product_count)
    return ScoringResult(tim_e=tim_e, cots=cots, modernization_recommendation=modernization)
