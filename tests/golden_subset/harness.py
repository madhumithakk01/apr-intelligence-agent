"""Golden-subset harness -- loading, scoring, and label bookkeeping.

The golden subset regression-gates every PR from this branch onward
(SPEC.md section 15). It comes in two halves, and the split is a
confidentiality requirement, not a convenience:

  committed fixture   tests/golden_subset/rows.json -- 20 invented rows.
                      No row, value, cost, or capability name is taken
                      or derived from the client workbook. This is what
                      CI runs, because nothing derived from the client
                      file goes to GitHub, de-identified or not.

  local fixture       tests/golden_subset/local/ -- de-identified rows
                      built from the client workbook by
                      build_local_subset.py. Gitignored, never pushed,
                      generated on demand by whoever has the file. When
                      present, the same suite runs over it too.

The invented fixture is built to exercise the code paths, not to
validate the parameters: it can catch a scoring regression, and it
cannot tell you whether 0.45/0.35/0.20 is the right weighting, because
its rows were chosen by the same person who would be checking. SPEC.md
section 12's empirical validation has to happen against the local
fixture (or the real portfolio), and its verdict travels as a label, not
as data -- see README.md.

No LLM call, no provider key: cost cells are converted deterministically
here, so a cell that would need cost_parsing's narrow LLM fallback is
recorded as withheld instead. CI runs this suite with no credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.scoring import kernel

FIXTURE_DIR = Path(__file__).resolve().parent
ROWS_PATH = FIXTURE_DIR / "rows.json"
LABELS_PATH = FIXTURE_DIR / "labels.json"

LOCAL_DIR = FIXTURE_DIR / "local"
LOCAL_ROWS_PATH = LOCAL_DIR / "rows.json"
LOCAL_LABELS_PATH = LOCAL_DIR / "labels.json"

MIN_ROWS = 20
MAX_ROWS = 25
"""SPEC.md section 13: ~20-25 rows. Small enough to label by hand and
keep labeled; large enough to cover the qualitative value space."""

ANALYST_LABEL_FIELDS = (
    "expected_time_decision",
    "expected_redundancy_typology",
    "expected_cots_action",
    "reviewer",
    "notes",
)

JUDGMENT_FIELDS = ("expected_time_decision", "expected_redundancy_typology", "expected_cots_action")
"""The subset of analyst fields that gate a PR once filled. `reviewer`
and `notes` are provenance, not expectations."""

IDENTITY_FIELDS = ("Owner", "Owner Email", "Application Description")
"""Absent from the committed fixture and redacted from the local one.
Asserted empty by the suite in both, so neither a careless local build
nor a hand edit can put an identity into a row that CI serializes."""

# Excel column -> kernel.ScoringInput field. Only the columns the kernel
# reads; the rest of the row is carried for the redundancy and cost
# stages that will use this same fixture on branches 9-11.
KERNEL_FIELD_MAP = {
    "Application ID": "application_id",
    "Application Name": "application_name",
    "Business Capability L2": "business_capability_l2",
    "Business Capability L3": "business_capability_l3",
    "Business Criticality": "business_criticality",
    "Strategic Relevance": "strategic_relevance",
    "Business Fitness": "business_fitness",
    "Usage & Adoption": "usage_adoption",
    "Application Stability": "application_stability",
    "Maintainability": "maintainability",
    "Availability": "availability",
    "Reliability": "reliability",
    "Scalability": "scalability",
    "Application Security Level": "application_security_level",
    "Skill availability": "skill_availability",
    "Functional redundancy": "functional_redundancy",
}

COST_FIELD_MAP = {
    "Annual FTE Cost": "annual_fte_cost",
    "Annual License Cost": "annual_license_cost",
    "Annual Infrastructure Cost": "annual_infrastructure_cost",
    "Other Costs": "other_costs",
}

MARKET_PRODUCT_COUNT_FIELD = "Market Product Count"
"""Fixture-only column, not a client column. Market retrieval is branch
12; this stands in for its output so the COTS-fit path and the section
12 COTS threshold are covered by the gate rather than left untested
until then. A row without it scores with a count of 0, which is what the
batch path passes today."""


def load_rows() -> List[Dict[str, Any]]:
    return json.loads(ROWS_PATH.read_text(encoding="utf-8"))


def load_labels() -> Dict[str, Any]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def has_local_fixture() -> bool:
    return LOCAL_ROWS_PATH.exists() and LOCAL_LABELS_PATH.exists()


def load_local_fixture() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The de-identified rows built from the client workbook, if this
    machine has them. ([], {}) everywhere else -- including CI."""
    if not has_local_fixture():
        return [], {}
    return (
        json.loads(LOCAL_ROWS_PATH.read_text(encoding="utf-8")),
        json.loads(LOCAL_LABELS_PATH.read_text(encoding="utf-8")),
    )


def _numeric_or_withheld(value: Any) -> Optional[float]:
    """Deterministic only -- see the module docstring on why this does
    not call cost_parsing's LLM fallback. Non-numeric text ("cannot
    disclose") is a withheld value, never a zero (SPEC.md section 2)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scoring_input(row: Dict[str, Any], market_product_count: Optional[int] = None) -> kernel.ScoringInput:
    kwargs: Dict[str, Any] = {field: row.get(column) for column, field in KERNEL_FIELD_MAP.items()}
    kwargs.update(
        {field: _numeric_or_withheld(row.get(column)) for column, field in COST_FIELD_MAP.items()}
    )
    if market_product_count is None:
        market_product_count = int(row.get(MARKET_PRODUCT_COUNT_FIELD) or 0)
    kwargs["market_product_count"] = market_product_count
    return kernel.ScoringInput(**kwargs)


def _aggregate(score: kernel.AggregateScore) -> Dict[str, Any]:
    return {
        "value": score.value,
        "status": score.status,
        "scored_axes": list(score.scored_axes),
        "unscored_axes": list(score.unscored_axes),
    }


def kernel_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    """The row's full deterministic scoring output, including which axes
    went unscored -- the unscored lists are the part that makes SPEC.md
    section 4 bug 1 (free-text values silently defaulting to 3) visible
    in a diff instead of invisible behind a plausible number."""
    result = kernel.score_application(scoring_input(row))
    return {
        "tim_e_score": result.tim_e.score,
        "tim_e_decision": result.tim_e.decision,
        "tim_e_raw_decision": result.tim_e.raw_decision,
        "floor_applied": result.tim_e.floor_applied,
        "security_classification": result.tim_e.security_classification,
        "value_score": _aggregate(result.tim_e.value_score),
        "health_score": _aggregate(result.tim_e.health_score),
        "consolidation_need": _aggregate(result.tim_e.consolidation_need),
        "cots_score": result.cots.score,
        "cots_recommendation": result.cots.recommendation,
        "cots_meets_threshold": result.cots.meets_threshold,
        "modernization_recommendation": result.modernization_recommendation,
    }


def blank_analyst_labels() -> Dict[str, Any]:
    return {field: None for field in ANALYST_LABEL_FIELDS}


def build_labels(
    rows: List[Dict[str, Any]],
    existing: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Regenerate snapshots for `rows`, carrying across any analyst
    labels already filled in, keyed by golden id."""
    previous = ((existing or {}).get("rows")) or {}
    labelled: Dict[str, Any] = {}
    for row in rows:
        golden_id = row["Application ID"]
        carried = (previous.get(golden_id) or {}).get("analyst_labels") or {}
        analyst = blank_analyst_labels()
        analyst.update({key: value for key, value in carried.items() if key in analyst})
        labelled[golden_id] = {"kernel_snapshot": kernel_snapshot(row), "analyst_labels": analyst}
    return {
        "fixture": "golden_subset",
        "note": note,
        "snapshot_note": (
            "kernel_snapshot is generated, not judged -- it records what app/scoring/kernel.py "
            "currently produces for each row, so an unintended scoring change fails a PR. "
            "analyst_labels are human judgments; they are filled by an internal reviewer, never "
            "generated."
        ),
        "rows": labelled,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analyst_labels(labels: Dict[str, Any], golden_id: str) -> Dict[str, Any]:
    return ((labels.get("rows") or {}).get(golden_id) or {}).get("analyst_labels") or {}


def filled_judgments(labels: Dict[str, Any], golden_id: str) -> Dict[str, Any]:
    """Only the judgment fields a reviewer actually filled in."""
    entry = analyst_labels(labels, golden_id)
    return {field: entry[field] for field in JUDGMENT_FIELDS if entry.get(field) is not None}


def label_coverage(labels: Dict[str, Any]) -> Dict[str, Any]:
    rows = labels.get("rows") or {}
    per_field = {
        field: sum(1 for golden_id in rows if analyst_labels(labels, golden_id).get(field) is not None)
        for field in JUDGMENT_FIELDS
    }
    return {
        "total_rows": len(rows),
        "rows_with_any_judgment": sum(1 for golden_id in rows if filled_judgments(labels, golden_id)),
        "per_field": per_field,
    }
