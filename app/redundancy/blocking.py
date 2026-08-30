"""Capability blocking -- CLAUDE.md sections 5 and 9.

Deterministic candidate grouping, generous on purpose: "a pairing missed
here can never be recovered downstream" (section 5). Blocking is O(n)
and portfolio-size-agnostic; within-cluster comparison (branch 10's
adjudicator) is O(k^2) per cluster, which is the stated, accepted cost
at current scale (section 9's scaling note).

The blocking key is (Business Capability L1, Business Capability L2) --
deliberately *not* L3. This is load-bearing, not an oversight: the
adjudicator's own typology (section 9) defines "Partial/Component
Overlap" as "L1/L2 match, L3 or description diverge." If blocking used
the full L1/L2/L3 path as its key, two applications with matching L1/L2
but a different L3 would land in different clusters and never reach the
adjudicator at all -- making that typology label unreachable. Blocking
must be coarser than the finest capability level specifically so the
adjudicator has L3 divergence left to detect.

Fallback hierarchy, generous by construction (CLAUDE.md section 9:
"falling back to L1 or Department if L2/L3 is missing... never exclude a
row from clustering entirely"):

  1. (L1, L2) both present -> group by the pair
  2. L1 present, L2 missing -> group by L1 alone
  3. L1 also missing, Department present -> group by Department
  4. Nothing usable at all -> one shared catch-all group, so a row is
     never dropped from clustering, only grouped as coarsely as the data
     allows

A cluster of size 1 has nothing to compare it against and is not
returned -- there is no redundancy question to ask about an application
with no capability (or Department) peers in this portfolio.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _normalize(value: Optional[Any]) -> Optional[str]:
    """None, and a blank/whitespace-only string, both mean "not usable
    for this tier of the key" -- collapsed to None so the fallback
    hierarchy below has one thing to check, not two."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _blocking_key(application: Dict[str, Any]) -> Tuple[str, ...]:
    l1 = _normalize(application.get("business_capability_l1"))
    l2 = _normalize(application.get("business_capability_l2"))
    department = _normalize(application.get("department"))

    if l1 and l2:
        return ("l1_l2", l1.casefold(), l2.casefold())
    if l1:
        return ("l1_only", l1.casefold())
    if department:
        return ("department", department.casefold())
    return ("unclassified",)


def _slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in text.upper())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "X"


_TIER_PREFIX = {
    "l1_l2": "L1L2",
    "l1_only": "L1",
    "department": "DEPT",
    "unclassified": "UNCLASSIFIED",
}


def _cluster_id(key: Tuple[str, ...]) -> str:
    tier, *parts = key
    prefix = _TIER_PREFIX[tier]
    if not parts:
        return f"CL-{prefix}"
    return f"CL-{prefix}-{'-'.join(_slug(part) for part in parts)}"


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    blocking_tier: str
    """Which fallback tier produced this cluster -- "l1_l2" | "l1_only" |
    "department" | "unclassified". Audit trail: a cluster built on a
    fallback tier is a weaker functional-match signal than one built on
    the primary (L1, L2) key, and the adjudicator (or a human reviewer)
    may want to weigh that."""
    application_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "blocking_tier": self.blocking_tier,
            "application_ids": list(self.application_ids),
        }


def block_by_capability(applications: List[Dict[str, Any]]) -> List[Cluster]:
    """Group applications into redundancy candidate clusters.

    Only clusters of 2+ members are returned -- a singleton has no peer
    to compare against, so branch 10's O(k^2) adjudication has nothing
    to do with it. Order is deterministic: clusters are returned in the
    order their key was first seen, and application_ids within a
    cluster preserve input order, so re-running blocking on the same
    input always produces the same output.
    """
    grouped: "OrderedDict[Tuple[str, ...], List[str]]" = OrderedDict()

    for application in applications:
        application_id = application.get("application_id")
        if not application_id:
            continue
        key = _blocking_key(application)
        grouped.setdefault(key, []).append(application_id)

    return [
        Cluster(cluster_id=_cluster_id(key), blocking_tier=key[0], application_ids=members)
        for key, members in grouped.items()
        if len(members) >= 2
    ]
