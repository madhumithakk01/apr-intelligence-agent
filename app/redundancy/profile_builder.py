"""Multi-axis profile building -- SPEC.md sections 5 and 9.

Deterministic. Replaces any single blended similarity score: each
application gets a profile of five axes kept *separate* -- "the
adjudicator sees these separately, never pre-blended" (section 9).
Building a profile here means assembling and lightly normalizing each
axis's raw material; comparing two profiles (deciding whether they
indicate a duplicate, a tiered overlap, or nothing) is branch 10's job,
not this module's.

  - Functional: capability tags + a deterministic description-similarity
    primitive (Jaccard over normalized tokens -- no LLM call; the
    section 5 table assigns this whole stage "Deterministic")
  - Scale/usage: FTE Count, Usage & Adoption, Business Criticality
  - Cost: normalized cost-per-FTE, never a raw annual total, which
    misleads across applications of very different scale
  - Risk/classification: Application Security Level (a data-
    classification label, not a health rating -- SPEC.md section 4 bug
    4), Application Stability, Availability
  - Technical: Technology Stack, Maintainability

Self-reported "Functional redundancy" is carried on the profile as its
own explicitly-flagged field, never folded into the functional axis or
any other -- SPEC.md section 9: "subject to owner self-preservation
bias -- never trusted as a standalone signal."

Every field here is None when the underlying value is missing or
withheld, exactly like every other stage in this system -- a caller
comparing two profiles must treat None as "unknown," never as a
guessable default (section 2). This module has no opinion on *why* a
value is None (blank in the source file, or gated out by disclosure
classification once that stage runs ahead of it in the full pipeline --
branch 6, not an ancestor of this one): a profile built from a raw
ingested dict and one built from a disclosure-gated dict behave
identically, which is what lets this module compose with either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.scoring.kernel import _aggregate_cost

_STOPWORDS = frozenset(
    "a an the and or of to for in on with by is are was were be been being "
    "this that these those as at from into over under it its".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tokenize(text: Optional[str]) -> Tuple[str, ...]:
    """Lowercase, split on non-alphanumeric runs, drop stopwords and
    empty tokens, dedupe, sort -- a stable, order-independent set ready
    for Jaccard comparison. Not a real NLP pipeline; a description-
    similarity signal only needs to be consistent and deterministic, not
    linguistically sophisticated."""
    if not text:
        return ()
    tokens = {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}
    return tuple(sorted(tokens))


@dataclass(frozen=True)
class FunctionalAxis:
    capability_l1: Optional[str]
    capability_l2: Optional[str]
    capability_l3: Optional[str]
    description_tokens: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability_l1": self.capability_l1,
            "capability_l2": self.capability_l2,
            "capability_l3": self.capability_l3,
            "description_tokens": list(self.description_tokens),
        }


@dataclass(frozen=True)
class ScaleUsageAxis:
    fte_count: Optional[int]
    usage_adoption: Optional[str]
    business_criticality: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fte_count": self.fte_count,
            "usage_adoption": self.usage_adoption,
            "business_criticality": self.business_criticality,
        }


@dataclass(frozen=True)
class CostAxis:
    cost_per_fte: Optional[float]
    """Total known annual cost / FTE Count (SPEC.md section 9: never a
    raw annual total). None if FTE Count is missing, zero, or every cost
    component is missing -- there is nothing to normalize."""
    is_complete: bool
    """False when cost_per_fte was computed from a partial cost sum (at
    least one of the four cost components missing) -- a caller comparing
    two applications' cost-per-FTE should weigh an incomplete figure as
    a weaker signal, never as equivalent to a fully-known one, and must
    never treat is_complete=False as license to guess the rest."""

    def as_dict(self) -> Dict[str, Any]:
        return {"cost_per_fte": self.cost_per_fte, "is_complete": self.is_complete}


@dataclass(frozen=True)
class RiskClassificationAxis:
    application_security_level: Optional[str]
    application_stability: Optional[str]
    availability: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "application_security_level": self.application_security_level,
            "application_stability": self.application_stability,
            "availability": self.availability,
        }


@dataclass(frozen=True)
class TechnicalAxis:
    technology_stack_tokens: Tuple[str, ...]
    maintainability: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "technology_stack_tokens": list(self.technology_stack_tokens),
            "maintainability": self.maintainability,
        }


@dataclass(frozen=True)
class ApplicationProfile:
    application_id: str
    functional: FunctionalAxis
    scale_usage: ScaleUsageAxis
    cost: CostAxis
    risk_classification: RiskClassificationAxis
    technical: TechnicalAxis
    functional_redundancy_self_report: Optional[str]
    """SPEC.md section 9: passed through as one input among several,
    never used standalone -- see the module docstring."""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "functional": self.functional.as_dict(),
            "scale_usage": self.scale_usage.as_dict(),
            "cost": self.cost.as_dict(),
            "risk_classification": self.risk_classification.as_dict(),
            "technical": self.technical.as_dict(),
            "functional_redundancy_self_report": self.functional_redundancy_self_report,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationProfile":
        """The reverse of as_dict() -- profiles cross the orchestration
        graph's checkpoint boundary as plain dicts (app.orchestration.state
        stores GraphState as JSON-serializable data), so a consumer that
        needs the typed dataclass back (the adjudicator, branch 10) uses
        this rather than re-deriving axis objects field by field."""
        functional = data["functional"]
        technical = data["technical"]
        return cls(
            application_id=data["application_id"],
            functional=FunctionalAxis(
                capability_l1=functional["capability_l1"],
                capability_l2=functional["capability_l2"],
                capability_l3=functional["capability_l3"],
                description_tokens=tuple(functional["description_tokens"]),
            ),
            scale_usage=ScaleUsageAxis(**data["scale_usage"]),
            cost=CostAxis(**data["cost"]),
            risk_classification=RiskClassificationAxis(**data["risk_classification"]),
            technical=TechnicalAxis(
                technology_stack_tokens=tuple(technical["technology_stack_tokens"]),
                maintainability=technical["maintainability"],
            ),
            functional_redundancy_self_report=data["functional_redundancy_self_report"],
        )


def _technology_stack_tokens(raw_stack: Optional[str]) -> Tuple[str, ...]:
    """Component-level tokens, not free-text tokens: "Dynamics 365,
    Azure SQL, Power Platform" splits into 3 components rather than
    being word-tokenized (which would scatter "365"/"SQL"/"Platform" as
    unrelated words and lose "which products" as the actual signal)."""
    if not raw_stack:
        return ()
    components = {
        component.strip().casefold()
        for component in re.split(r"[,/;]| and ", raw_stack)
        if component.strip()
    }
    return tuple(sorted(components))


def _fte_count(raw_value: Optional[Any]) -> Optional[int]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        return int(raw_value)
    return None


def _cost_axis(application: Dict[str, Any]) -> CostAxis:
    fte_count = _fte_count(application.get("fte_count"))
    aggregate = _aggregate_cost(
        application.get("annual_fte_cost"),
        application.get("annual_license_cost"),
        application.get("annual_infrastructure_cost"),
        application.get("other_costs"),
    )
    if fte_count is None or fte_count <= 0 or aggregate.total_known is None:
        return CostAxis(cost_per_fte=None, is_complete=False)
    return CostAxis(
        cost_per_fte=round(aggregate.total_known / fte_count, 2),
        is_complete=aggregate.is_complete,
    )


def build_profile(application: Dict[str, Any]) -> ApplicationProfile:
    """One application dict -> one ApplicationProfile. Accepts either a
    raw ingested application or a disclosure-gated one (see module
    docstring) -- the only contract is the canonical snake_case field
    names app.ingestion.row_mapping already produces."""
    functional = FunctionalAxis(
        capability_l1=_normalize(application.get("business_capability_l1")),
        capability_l2=_normalize(application.get("business_capability_l2")),
        capability_l3=_normalize(application.get("business_capability_l3")),
        description_tokens=_tokenize(_normalize(application.get("application_description"))),
    )
    scale_usage = ScaleUsageAxis(
        fte_count=_fte_count(application.get("fte_count")),
        usage_adoption=_normalize(application.get("usage_adoption")),
        business_criticality=_normalize(application.get("business_criticality")),
    )
    cost = _cost_axis(application)
    risk_classification = RiskClassificationAxis(
        application_security_level=_normalize(application.get("application_security_level")),
        application_stability=_normalize(application.get("application_stability")),
        availability=_normalize(application.get("availability")),
    )
    technical = TechnicalAxis(
        technology_stack_tokens=_technology_stack_tokens(_normalize(application.get("technology_stack"))),
        maintainability=_normalize(application.get("maintainability")),
    )
    return ApplicationProfile(
        application_id=application.get("application_id"),
        functional=functional,
        scale_usage=scale_usage,
        cost=cost,
        risk_classification=risk_classification,
        technical=technical,
        functional_redundancy_self_report=_normalize(application.get("functional_redundancy")),
    )


def build_profiles(applications: List[Dict[str, Any]]) -> Dict[str, ApplicationProfile]:
    """Batch convenience: application_id -> ApplicationProfile, skipping
    any row with no application_id (nothing to key it by)."""
    profiles: Dict[str, ApplicationProfile] = {}
    for application in applications:
        application_id = application.get("application_id")
        if not application_id:
            continue
        profiles[application_id] = build_profile(application)
    return profiles


# --- comparison primitives ---------------------------------------------
# Not full pairwise adjudication (branch 10's job) -- deterministic
# building blocks the adjudicator can call rather than re-deriving.


def description_similarity(a: ApplicationProfile, b: ApplicationProfile) -> Optional[float]:
    """Jaccard index over normalized description tokens: |intersection|
    / |union|, in [0.0, 1.0]. None if either application's description
    is missing/withheld entirely -- an absent description is not the
    same as a description with zero token overlap, and must not be
    scored as "0% similar" (a false, misleadingly confident signal)."""
    tokens_a, tokens_b = set(a.functional.description_tokens), set(b.functional.description_tokens)
    if not tokens_a or not tokens_b:
        return None
    union = tokens_a | tokens_b
    if not union:
        return None
    return round(len(tokens_a & tokens_b) / len(union), 4)


CAPABILITY_MATCH_FULL = "full"
CAPABILITY_MATCH_PARTIAL = "partial"
CAPABILITY_MATCH_SUPERFICIAL = "superficial"


def capability_match_level(a: ApplicationProfile, b: ApplicationProfile) -> str:
    """SPEC.md section 9's typology is phrased directly in terms of
    which capability levels match -- this is that comparison, made once
    so the adjudicator doesn't re-derive it per pair:

      full: L1, L2, and L3 all match (candidate: True Duplicate or
            Scale-Tiered Overlap, per the adjudicator's other axes)
      partial: L1 and L2 match, L3 diverges (candidate: Partial/
               Component Overlap)
      superficial: anything else -- including two applications that
                   only share this blocking cluster via a coarser
                   fallback tier (Department, or no capability data at
                   all), where L1 itself may not even match
    """
    l1_match = bool(a.functional.capability_l1) and _fold(a.functional.capability_l1) == _fold(
        b.functional.capability_l1
    )
    l2_match = bool(a.functional.capability_l2) and _fold(a.functional.capability_l2) == _fold(
        b.functional.capability_l2
    )
    l3_match = bool(a.functional.capability_l3) and _fold(a.functional.capability_l3) == _fold(
        b.functional.capability_l3
    )
    if l1_match and l2_match and l3_match:
        return CAPABILITY_MATCH_FULL
    if l1_match and l2_match:
        return CAPABILITY_MATCH_PARTIAL
    return CAPABILITY_MATCH_SUPERFICIAL


def _fold(value: Optional[str]) -> Optional[str]:
    return value.casefold() if value is not None else None
