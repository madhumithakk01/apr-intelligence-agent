"""app.orchestration.nodes.ingest -- the graph's real loader wiring.

Mirrors tests/test_import_dataset_collisions.py's fixtures and its
discipline of asserting no LLM call happens for clean/refusal-only
workbooks (app.ingestion.cost_parsing's narrow fallback is exercised
elsewhere, not incidentally here).
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion import cost_parsing
from app.orchestration.nodes import ingest

REQUIRED_COLUMNS = [
    "Application ID", "Application Name", "Owner", "Owner Email", "Department",
    "Application Description", "Application Status", "Business Criticality",
    "Business Fitness", "Strategic Relevance", "Usage & Adoption",
    "Functional redundancy", "Application Security Level", "Maintainability",
    "Application Stability", "Skill availability", "Availability", "Reliability",
    "Scalability", "Technology Stack", "Annual FTE Cost", "Annual License Cost",
    "FTE Count", "Annual Infrastructure Cost", "Other Costs",
    "Business Capability L1", "Business Capability L2", "Business Capability L3",
]


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("get_completion should not be called for these clean/refusal-only fixtures")

    monkeypatch.setattr(cost_parsing, "get_completion", _fail)


def _row(**overrides):
    base = {col: "N/A" for col in REQUIRED_COLUMNS}
    base.update(
        {
            "Annual FTE Cost": 100000,
            "Annual License Cost": 50000,
            "FTE Count": 5,
            "Annual Infrastructure Cost": 20000,
            "Other Costs": 1000,
        }
    )
    base.update(overrides)
    return base


def _write_workbook(path, sheet1_rows, sheet2_rows=None):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(sheet1_rows).to_excel(writer, sheet_name="Sheet1", index=False)
        if sheet2_rows is not None:
            pd.DataFrame(sheet2_rows).to_excel(writer, sheet_name="Sheet2", index=False)


def test_applications_already_in_state_pass_through_unchanged_even_with_a_path_set(tmp_path):
    workbook = tmp_path / "unused.xlsx"
    _write_workbook(workbook, sheet1_rows=[_row(**{"Application ID": "APP-999"})])

    supplied = [{"application_id": "SYN-1"}]
    result = ingest({"applications": supplied, "dataset_path": str(workbook)})

    assert result["applications"] == supplied
    assert "ingestion_collisions" not in result


def test_no_applications_and_no_path_yields_empty_list():
    result = ingest({})
    assert result["applications"] == []


def test_loads_and_maps_rows_from_the_dataset_path(tmp_path):
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[
            _row(**{"Application ID": "APP-001", "Application Name": "Widget Tracker"}),
            _row(**{"Application ID": "APP-002", "Application Name": "Ledger Core"}),
        ],
    )

    result = ingest({"dataset_path": str(workbook)})

    applications = result["applications"]
    assert {app["application_id"] for app in applications} == {"APP-001", "APP-002"}
    widget = next(app for app in applications if app["application_id"] == "APP-001")
    assert widget["application_name"] == "Widget Tracker"
    assert widget["annual_fte_cost"] == 100000.0
    assert widget["fte_count"] == 5
    assert result["ingestion_collisions"] == []


def test_colliding_application_ids_are_excluded_and_surfaced_not_dropped_silently(tmp_path):
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[_row(**{"Application ID": "APP-001", "Application Name": "First"})],
        sheet2_rows=[
            _row(**{"Application ID": "APP-001", "Application Name": "Second"}),
            _row(**{"Application ID": "APP-002", "Application Name": "Clean"}),
        ],
    )

    result = ingest({"dataset_path": str(workbook)})

    assert {app["application_id"] for app in result["applications"]} == {"APP-002"}
    assert len(result["ingestion_collisions"]) == 1
    assert result["ingestion_collisions"][0]["application_id"] == "APP-001"
    assert len(result["ingestion_collisions"][0]["occurrences"]) == 2


def test_refusal_cost_text_does_not_crash_ingest(tmp_path):
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[_row(**{"Application ID": "APP-001", "Other Costs": "cannot disclose"})],
    )

    result = ingest({"dataset_path": str(workbook)})

    application = result["applications"][0]
    assert application["other_costs"] is None
    assert "cannot disclose" in application["numeric_field_notes"]


def test_stage_log_records_ingest_as_complete():
    result = ingest({})
    entry = result["stage_log"][0]
    assert entry["stage"] == "ingest"
    assert entry["status"] == "complete"
