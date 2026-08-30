"""One raw Excel row -> one ApplicationInput-shaped dict.

Single source of truth for the raw-column -> canonical-field mapping
(CLAUDE.md section 1: no invented second copy). Before this module
existed, import_dataset.py hand-built this mapping inline as kwargs to
the `Application` ORM model; app/orchestration/nodes.ingest needs the
identical mapping to build application dicts for the graph, which is
what this extraction is for. Both callers get the same safe cost
parsing (CLAUDE.md section 4 bug 6: never crash, never silently coerce
a withheld cost to zero) and the same collision-surfacing contract from
whichever caller drives ExcelLoader.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.ingestion.cost_parsing import build_numeric_field_notes, parse_cost_cell, parse_fte_count

# canonical field -> raw Excel column header. Order matches
# app/schemas.py ApplicationInput and the Application ORM model.
STRING_FIELD_COLUMNS: Dict[str, str] = {
    "application_name": "Application Name",
    "owner": "Owner",
    "owner_email": "Owner Email",
    "department": "Department",
    "application_description": "Application Description",
    "application_status": "Application Status",
    "business_criticality": "Business Criticality",
    "business_fitness": "Business Fitness",
    "strategic_relevance": "Strategic Relevance",
    "usage_adoption": "Usage & Adoption",
    "functional_redundancy": "Functional redundancy",
    "application_security_level": "Application Security Level",
    "maintainability": "Maintainability",
    "application_stability": "Application Stability",
    "skill_availability": "Skill availability",
    "availability": "Availability",
    "reliability": "Reliability",
    "scalability": "Scalability",
    "technology_stack": "Technology Stack",
    "business_capability_l1": "Business Capability L1",
    "business_capability_l2": "Business Capability L2",
    "business_capability_l3": "Business Capability L3",
}

COST_FIELD_COLUMNS: Dict[str, str] = {
    "annual_fte_cost": "Annual FTE Cost",
    "annual_license_cost": "Annual License Cost",
    "annual_infrastructure_cost": "Annual Infrastructure Cost",
    "other_costs": "Other Costs",
}

FTE_COUNT_COLUMN = "FTE Count"

REQUIRED_COLUMNS = (
    ["Application ID"]
    + list(STRING_FIELD_COLUMNS.values())
    + list(COST_FIELD_COLUMNS.values())
    + [FTE_COUNT_COLUMN]
)


def build_application_fields(row: "pd.Series", *, application_id: str) -> Dict[str, Any]:
    """One row -> one ApplicationInput-shaped dict, safely.

    `application_id` is taken as given rather than re-derived from
    `row["Application ID"]` -- callers already compute it once (stripped,
    stringified) to use as the parse-error log key and the collision
    key, and this keeps both uses referring to literally the same value.
    """
    fields: Dict[str, Any] = {"application_id": application_id}
    for canonical, column in STRING_FIELD_COLUMNS.items():
        fields[canonical] = row.get(column)

    parsed = {
        canonical: parse_cost_cell(row.get(column), field_name=canonical, application_id=application_id)
        for canonical, column in COST_FIELD_COLUMNS.items()
    }
    parsed["fte_count"] = parse_fte_count(row.get(FTE_COUNT_COLUMN), application_id=application_id)

    for canonical in COST_FIELD_COLUMNS:
        fields[canonical] = parsed[canonical].value
    fte_value = parsed["fte_count"].value
    fields["fte_count"] = int(round(fte_value)) if fte_value is not None else None

    fields["numeric_field_notes"] = build_numeric_field_notes(parsed)
    return fields
