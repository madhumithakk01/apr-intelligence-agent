"""app.ingestion.row_mapping -- the shared raw-row -> ApplicationInput
mapping behind both import_dataset.py and app.orchestration.nodes.ingest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion import cost_parsing
from app.ingestion.row_mapping import build_application_fields


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("get_completion should not be called for these clean/refusal-only fixtures")

    monkeypatch.setattr(cost_parsing, "get_completion", _fail)


def _row(**overrides):
    base = {
        "Application ID": "APP-1",
        "Application Name": "Widget Tracker",
        "Owner": "A. Person",
        "Owner Email": "a.person@example.com",
        "Department": "Ops",
        "Application Description": "Tracks widgets",
        "Application Status": "Production",
        "Business Criticality": "Strategic",
        "Business Fitness": "Fully supports",
        "Strategic Relevance": "Directly supports",
        "Usage & Adoption": "High",
        "Functional redundancy": "Unique",
        "Application Security Level": "Confidential",
        "Maintainability": "Simple",
        "Application Stability": "Stable",
        "Skill availability": "Well supported",
        "Availability": "Always available",
        "Reliability": "Very reliable",
        "Scalability": "Highly scalable",
        "Technology Stack": "Django, Postgres",
        "Annual FTE Cost": 100000,
        "Annual License Cost": 50000,
        "FTE Count": 5,
        "Annual Infrastructure Cost": 20000,
        "Other Costs": 1000,
        "Business Capability L1": "Ops",
        "Business Capability L2": "Track to Resolve",
        "Business Capability L3": "Widget Tracking",
    }
    base.update(overrides)
    return pd.Series(base)


def test_maps_every_string_field_verbatim():
    fields = build_application_fields(_row(), application_id="APP-1")

    assert fields["application_id"] == "APP-1"
    assert fields["application_name"] == "Widget Tracker"
    assert fields["owner"] == "A. Person"
    assert fields["owner_email"] == "a.person@example.com"
    assert fields["department"] == "Ops"
    assert fields["application_description"] == "Tracks widgets"
    assert fields["business_criticality"] == "Strategic"
    assert fields["technology_stack"] == "Django, Postgres"
    assert fields["business_capability_l3"] == "Widget Tracking"


def test_parses_clean_costs_and_fte_count():
    fields = build_application_fields(_row(), application_id="APP-1")

    assert fields["annual_fte_cost"] == 100000.0
    assert fields["annual_license_cost"] == 50000.0
    assert fields["annual_infrastructure_cost"] == 20000.0
    assert fields["other_costs"] == 1000.0
    assert fields["fte_count"] == 5
    assert fields["numeric_field_notes"] is None


def test_uses_the_given_application_id_not_the_rows_own_column():
    """The caller computes application_id once (stripped, deduped
    against collisions) and this function must not silently re-derive a
    different value from the row."""
    fields = build_application_fields(_row(**{"Application ID": "  APP-1  "}), application_id="APP-1")
    assert fields["application_id"] == "APP-1"


def test_refusal_cost_text_is_withheld_not_a_crash():
    fields = build_application_fields(_row(**{"Other Costs": "cannot disclose"}), application_id="APP-1")
    assert fields["other_costs"] is None
    assert fields["numeric_field_notes"] is not None
    assert "cannot disclose" in fields["numeric_field_notes"]


def test_non_finite_fte_count_never_raises():
    fields = build_application_fields(_row(**{"FTE Count": "Infinity"}), application_id="APP-1")
    assert fields["fte_count"] is None
    assert "fte_count" in fields["numeric_field_notes"]


def test_missing_column_yields_none_rather_than_a_keyerror():
    row = _row()
    del row["Department"]
    fields = build_application_fields(row, application_id="APP-1")
    assert fields["department"] is None
