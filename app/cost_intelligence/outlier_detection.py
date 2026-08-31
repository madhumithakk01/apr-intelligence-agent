"""Cost outlier detection -- SPEC.md sections 5, 9, 12.

Deterministic statistics decide the flag; nothing here calls an LLM
(that is explainability.py's job, and only for a row already flagged
here). Peer grouping reuses the same capability clusters redundancy
blocking already produces (app.redundancy.blocking) -- "peer cluster"
in SPEC.md section 12's MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER is the
same concept as a blocking cluster, and building a second, parallel
notion of "similar applications" alongside blocking's would be exactly
the kind of second copy SPEC.md section 1 rules out. The comparison
metric is cost-per-FTE (app.redundancy.profile_builder's CostAxis) --
never a raw annual total, which misleads across applications of very
different scale, the same normalization principle SPEC.md section 9
states for redundancy comparison.

The statistical rule is Tukey's IQR fence (flag outside
[Q1 - k*IQR, Q3 + k*IQR]), not a z-score/standard-deviation rule: IQR is
robust to the small peer-cluster sizes this system actually has (the
floor is 5) and to the very outliers it is trying to detect, which would
otherwise inflate the standard deviation a z-score depends on.

A cluster with fewer than MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER members
with a *known* cost-per-FTE never flags anything from that cluster --
SPEC.md section 12: "refuse to flag rather than manufacture a signal
from noise." A member whose own cost is unknown is never flagged either
(there is nothing to compare) and does not count toward the floor.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.redundancy.profile_builder import ApplicationProfile
from app.scoring import governance_params as gp

DIRECTION_HIGH = "high"
DIRECTION_LOW = "low"


@dataclass(frozen=True)
class ClusterCostStats:
    cluster_id: str
    peer_count: int
    """Number of members with a known cost-per-FTE -- the figure
    MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER is measured against, not the
    cluster's raw membership count."""
    median: float
    q1: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "peer_count": self.peer_count,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "lower_fence": self.lower_fence,
            "upper_fence": self.upper_fence,
        }


@dataclass(frozen=True)
class CostOutlierFlag:
    application_id: str
    cluster_id: str
    cost_per_fte: float
    direction: str  # DIRECTION_HIGH | DIRECTION_LOW
    cluster_stats: ClusterCostStats

    def as_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "cluster_id": self.cluster_id,
            "cost_per_fte": self.cost_per_fte,
            "direction": self.direction,
            "cluster_stats": self.cluster_stats.as_dict(),
        }


def _quartiles(values: List[float]) -> "tuple[float, float, float]":
    """(Q1, median, Q3) via Tukey's original hinges -- the median of the
    lower half and the median of the upper half
    (statistics.quantiles(..., method="inclusive")) -- not linear
    interpolation (method="exclusive"). This is not a stylistic choice:
    at the small peer-cluster sizes this system actually runs at (the
    floor is 5), the interpolation method places Q3 partway *between*
    the second-highest value and the outlier itself, which inflates the
    IQR fence by the outlier's own magnitude and can make a genuine
    outlier undetectable -- the inclusive method's hinges are always
    real data points, immune to that self-defeating effect."""
    if len(values) == 1:
        only = values[0]
        return only, only, only
    ordered = sorted(values)
    median = statistics.median(ordered)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return quartiles[0], median, quartiles[2]


def _cluster_cost_stats(cluster_id: str, cost_values: List[float]) -> Optional[ClusterCostStats]:
    if len(cost_values) < gp.MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER:
        return None
    q1, median, q3 = _quartiles(cost_values)
    iqr = q3 - q1
    k = gp.COST_OUTLIER_IQR_MULTIPLIER
    return ClusterCostStats(
        cluster_id=cluster_id,
        peer_count=len(cost_values),
        median=median,
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower_fence=q1 - k * iqr,
        upper_fence=q3 + k * iqr,
    )


def detect_cluster_outliers(
    cluster_id: str, member_profiles: List[ApplicationProfile]
) -> List[CostOutlierFlag]:
    """Every member of one cluster whose cost-per-FTE falls outside the
    cluster's own IQR fence. [] if the cluster has fewer than
    MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER members with a known cost."""
    known = [
        (profile.application_id, profile.cost.cost_per_fte)
        for profile in member_profiles
        if profile.cost.cost_per_fte is not None
    ]
    stats = _cluster_cost_stats(cluster_id, [cost for _application_id, cost in known])
    if stats is None:
        return []

    flags = []
    for application_id, cost in known:
        if cost > stats.upper_fence:
            flags.append(CostOutlierFlag(application_id, cluster_id, cost, DIRECTION_HIGH, stats))
        elif cost < stats.lower_fence:
            flags.append(CostOutlierFlag(application_id, cluster_id, cost, DIRECTION_LOW, stats))
    return flags


def detect_cost_outliers(
    clusters: List[Dict[str, Any]], profiles: Dict[str, ApplicationProfile]
) -> List[CostOutlierFlag]:
    """clusters: the blocking-cluster dicts (app.redundancy.blocking's
    Cluster.as_dict() shape, or anything with "cluster_id" and
    "application_ids"). profiles: application_id -> ApplicationProfile
    (app.redundancy.profile_builder), the same shape build_profiles
    already produces."""
    flags: List[CostOutlierFlag] = []
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "unknown")
        member_profiles = [
            profiles[application_id]
            for application_id in (cluster.get("application_ids") or [])
            if application_id in profiles
        ]
        flags.extend(detect_cluster_outliers(cluster_id, member_profiles))
    return flags
