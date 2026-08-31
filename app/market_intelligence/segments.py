"""Redundancy-surviving segment construction -- SPEC.md sections 8, 9.

Deterministic, and deliberately not one of the three files SPEC.md
section 13 names for this package (graph.py, tools.py, extraction.py) --
this logic earns its own module because it is a distinct responsibility
(deciding *what* the agent researches) from both the adjudication policy
that feeds it (app.redundancy.recommendation_policy) and the agent loop
that consumes its output (graph.py).

SPEC.md section 8: the agent fans out "once per redundancy-surviving
segment -- not once per raw app; segments come from the typology in
section 9, so a Scale-Tiered Overlap produces two differently-framed
research targets from one cluster." Section 9's typology table states
each typology's market-research cardinality explicitly:

  True Duplicate            -> once, for the retained application
  Scale-Tiered Overlap      -> separately per tier, tier-framed
  Partial/Component Overlap -> separately, one segment per application
  Distinct                  -> individually, one segment per application
  Indeterminate/Adjudication Failed -> deferred, no segment at all

An application that never shared a blocking cluster with any peer
(app.redundancy.blocking drops singleton clusters) never reaches the
adjudicator at all and so has no verdict to read a cardinality from --
but section 8's own stated purpose (COTS discovery across the portfolio)
does not stop at the redundant subset, and section 9's "Distinct"
cardinality ("individually") is exactly the right treatment for a
capability with no peer to compare against either. Such an application
gets its own standalone segment too; see build_segments's second pass.

"Which of two comparable applications is the retained one" (True
Duplicate) and "which tier is the heavier platform" (Scale-Tiered
Overlap) both reuse the same FTE-count-then-cost-per-FTE tie-break
app.redundancy.recommendation_policy already uses to decide the same
question for its own gates -- one rule for "which application is
heavier," not a second copy of it here.

Query text is built only from capability labels and technology stack --
never application names, owner/email, or cost figures -- since it is
sent to an external search API (app.market_intelligence.tools), not an
LLM call gated by SPEC.md section 11's DataSensitivity rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

STANDALONE = "standalone"
TIER_ENTERPRISE = "tier_enterprise"
TIER_LIGHT = "tier_light"
PARTIAL_OVERLAP = "partial_overlap"

_TRUE_DUPLICATE = "True Duplicate"
_SCALE_TIERED_OVERLAP = "Scale-Tiered Overlap"
_PARTIAL_COMPONENT_OVERLAP = "Partial/Component Overlap"
_DISTINCT = "Distinct"
_DEFERRED_TYPOLOGIES = {"Indeterminate — Withheld Data", "Adjudication Failed"}
"""String literals, not imported from app.redundancy.adjudicator: this
module only needs to recognize these five typology names as they appear
in the already-serialized verdict dicts crossing the checkpoint boundary
(state["verdicts"]), the same way every other consumer of that shape
(e.g. app.orchestration.nodes) reads it as plain data rather than
reconstructing adjudicator types."""

# Processed before Indeterminate/Adjudication Failed so a real verdict
# always "claims" an application ahead of a deferred one for the same
# app in a different pairing within the same (3+-member) cluster.
_TYPOLOGY_PRIORITY = {
    _TRUE_DUPLICATE: 0,
    _SCALE_TIERED_OVERLAP: 1,
    _PARTIAL_COMPONENT_OVERLAP: 2,
    _DISTINCT: 3,
}

_FRAMING_QUERY_TEMPLATES = {
    STANDALONE: "{capability} software alternatives",
    TIER_ENTERPRISE: "enterprise-grade {capability} platforms",
    TIER_LIGHT: "lightweight {capability} tools for small teams",
    PARTIAL_OVERLAP: "{capability} software vendors",
}


@dataclass(frozen=True)
class Segment:
    segment_id: str
    application_id: str
    cluster_id: Optional[str]
    typology: Optional[str]
    framing: str
    capability_label: str
    seed_query: str
    self_match_terms: Tuple[str, ...]
    """The application's own name and technology-stack components --
    never used in seed_query (which must not leak client-specific
    product names into an external search), but needed by the agent's
    self-match filter (SPEC.md section 8) to recognize when a search
    result is naming the client's own system rather than a competitor."""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "application_id": self.application_id,
            "cluster_id": self.cluster_id,
            "typology": self.typology,
            "framing": self.framing,
            "capability_label": self.capability_label,
            "seed_query": self.seed_query,
            "self_match_terms": list(self.self_match_terms),
        }


def _capability_label(application: Dict[str, Any]) -> str:
    for field in ("business_capability_l3", "business_capability_l2", "business_capability_l1"):
        value = application.get(field)
        if value:
            return str(value).strip()
    return "this application's capability"


def _seed_query(application: Dict[str, Any], framing: str) -> str:
    capability = _capability_label(application)
    template = _FRAMING_QUERY_TEMPLATES[framing]
    return template.format(capability=capability)


def _fte_count(profile: Optional[Dict[str, Any]]) -> Optional[float]:
    if not profile:
        return None
    return (profile.get("scale_usage") or {}).get("fte_count")


def _cost_per_fte(profile: Optional[Dict[str, Any]]) -> Optional[float]:
    if not profile:
        return None
    return (profile.get("cost") or {}).get("cost_per_fte")


def _self_match_terms(application: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
    terms = []
    name = application.get("application_name")
    if name:
        terms.append(str(name).strip())
    stack_tokens = (profile or {}).get("technical", {}).get("technology_stack_tokens") or []
    terms.extend(str(token) for token in stack_tokens)
    return tuple(dict.fromkeys(term for term in terms if term))  # dedupe, preserve order


def _heavier_lighter(
    application_id_a: str, application_id_b: str, profiles: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """(heavier, lighter) application_id, by FTE count then cost-per-FTE,
    falling back to input order -- the same tie-break
    app.redundancy.recommendation_policy._heavier_lighter uses, applied
    here to the serialized profile dicts already in GraphState rather
    than to ApplicationProfile objects, since this module has no other
    reason to deserialize them."""
    profile_a, profile_b = profiles.get(application_id_a), profiles.get(application_id_b)
    fte_a, fte_b = _fte_count(profile_a), _fte_count(profile_b)
    if fte_a is not None and fte_b is not None and fte_a != fte_b:
        return (application_id_a, application_id_b) if fte_a > fte_b else (application_id_b, application_id_a)
    cost_a, cost_b = _cost_per_fte(profile_a), _cost_per_fte(profile_b)
    if cost_a is not None and cost_b is not None and cost_a != cost_b:
        return (application_id_a, application_id_b) if cost_a > cost_b else (application_id_b, application_id_a)
    return application_id_a, application_id_b


def build_segments(
    verdicts: List[Dict[str, Any]],
    applications: List[Dict[str, Any]],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Segment]:
    applications_by_id = {
        application["application_id"]: application
        for application in applications
        if application.get("application_id")
    }

    segments: List[Segment] = []
    covered: set = set()
    deferred: set = set()

    def _add(application_id: str, framing: str, cluster_id: Optional[str], typology: Optional[str]) -> None:
        if application_id in covered or application_id not in applications_by_id:
            return
        application = applications_by_id[application_id]
        segments.append(
            Segment(
                segment_id=f"SEG-{application_id}-{framing}",
                application_id=application_id,
                cluster_id=cluster_id,
                typology=typology,
                framing=framing,
                capability_label=_capability_label(application),
                seed_query=_seed_query(application, framing),
                self_match_terms=_self_match_terms(application, profiles.get(application_id)),
            )
        )
        covered.add(application_id)

    for verdict in sorted(verdicts, key=lambda v: _TYPOLOGY_PRIORITY.get(v.get("typology"), 4)):
        typology = verdict.get("typology")
        cluster_id = verdict.get("cluster_id")
        a, b = verdict.get("application_id_a"), verdict.get("application_id_b")
        if not a or not b:
            continue

        if typology == _TRUE_DUPLICATE:
            retained, retired = _heavier_lighter(a, b, profiles)
            _add(retained, STANDALONE, cluster_id, typology)
            covered.add(retired)  # explicitly never researched -- section 9
        elif typology == _SCALE_TIERED_OVERLAP:
            heavier, lighter = _heavier_lighter(a, b, profiles)
            _add(heavier, TIER_ENTERPRISE, cluster_id, typology)
            _add(lighter, TIER_LIGHT, cluster_id, typology)
        elif typology == _PARTIAL_COMPONENT_OVERLAP:
            _add(a, PARTIAL_OVERLAP, cluster_id, typology)
            _add(b, PARTIAL_OVERLAP, cluster_id, typology)
        elif typology == _DISTINCT:
            _add(a, STANDALONE, cluster_id, typology)
            _add(b, STANDALONE, cluster_id, typology)
        elif typology in _DEFERRED_TYPOLOGIES:
            for application_id in (a, b):
                if application_id not in covered:
                    deferred.add(application_id)

    # Applications with no cluster at all (blocking dropped them as
    # singletons) never reached the adjudicator and so triggered none of
    # the branches above -- section 9's "Distinct" cardinality is the
    # right treatment: researched individually, on its own.
    for application_id in applications_by_id:
        if application_id not in covered and application_id not in deferred:
            _add(application_id, STANDALONE, None, None)

    return segments
