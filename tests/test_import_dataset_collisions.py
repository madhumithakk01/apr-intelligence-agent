import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.database.db as db_module
import app.services.import_service as import_service_module
import import_dataset
from app.database.models import Application
from app.ingestion import cost_parsing

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


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("get_completion should not be called for these clean/refusal-only fixtures")

    monkeypatch.setattr(cost_parsing, "get_completion", _fail)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_apr.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(import_service_module, "SessionLocal", session_local)
    monkeypatch.setattr(import_dataset, "engine", engine)

    return engine, session_local


def test_intra_file_collision_excluded_from_import(tmp_path, temp_db):
    engine, session_local = temp_db
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[_row(**{"Application ID": "APP-001", "Application Name": "SalesHub CRM"})],
        sheet2_rows=[
            _row(**{"Application ID": "APP-001", "Application Name": "Global ERP Core"}),
            _row(**{"Application ID": "APP-002", "Application Name": "Payroll System"}),
        ],
    )

    summary = import_dataset.run(dataset_path=str(workbook))

    assert summary.imported == 1
    assert len(summary.collisions) == 1
    assert summary.collisions[0]["application_id"] == "APP-001"
    assert summary.collisions[0]["source"] == "duplicate-in-file"

    session = session_local()
    try:
        assert session.query(Application).filter_by(application_id="APP-001").first() is None
        assert session.query(Application).filter_by(application_id="APP-002").first() is not None
    finally:
        session.close()


def test_existing_in_db_collision_not_overwritten(tmp_path, temp_db):
    engine, session_local = temp_db
    from app.database.db import Base

    Base.metadata.create_all(bind=engine)
    session = session_local()
    try:
        session.add(Application(application_id="APP-001", application_name="Original Name"))
        session.commit()
    finally:
        session.close()

    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[_row(**{"Application ID": "APP-001", "Application Name": "Corrected Name"})],
    )

    summary = import_dataset.run(dataset_path=str(workbook))

    assert summary.imported == 0
    assert len(summary.collisions) == 1
    assert summary.collisions[0]["source"] == "existing-in-db"

    session = session_local()
    try:
        row = session.query(Application).filter_by(application_id="APP-001").first()
        assert row.application_name == "Original Name"
    finally:
        session.close()


def test_clean_import_no_collisions(tmp_path, temp_db):
    engine, session_local = temp_db
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[
            _row(**{"Application ID": "APP-001", "Application Name": "SalesHub CRM"}),
            _row(**{"Application ID": "APP-002", "Application Name": "Payroll System"}),
        ],
    )

    summary = import_dataset.run(dataset_path=str(workbook))

    assert summary.imported == 2
    assert summary.collisions == []

    session = session_local()
    try:
        row = session.query(Application).filter_by(application_id="APP-001").first()
        assert row.annual_fte_cost == 100000.0
        assert row.numeric_field_notes is None
    finally:
        session.close()


def test_refusal_cost_text_does_not_crash_import(tmp_path, temp_db):
    engine, session_local = temp_db
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[
            _row(
                **{
                    "Application ID": "APP-001",
                    "Application Name": "SalesHub CRM",
                    "Other Costs": "cannot disclose",
                }
            )
        ],
    )

    summary = import_dataset.run(dataset_path=str(workbook))

    assert summary.imported == 1
    session = session_local()
    try:
        row = session.query(Application).filter_by(application_id="APP-001").first()
        assert row.other_costs is None
        assert row.numeric_field_notes is not None
        assert "cannot disclose" in row.numeric_field_notes
    finally:
        session.close()


def test_non_finite_fte_count_does_not_crash_import(tmp_path, temp_db):
    """int(round(...)) on an inf/nan FTE Count value used to raise
    OverflowError/ValueError and crash the whole batch."""
    engine, session_local = temp_db
    workbook = tmp_path / "dataset.xlsx"
    _write_workbook(
        workbook,
        sheet1_rows=[
            _row(**{"Application ID": "APP-001", "Application Name": "SalesHub CRM", "FTE Count": "Infinity"})
        ],
    )

    summary = import_dataset.run(dataset_path=str(workbook))

    assert summary.imported == 1
    session = session_local()
    try:
        row = session.query(Application).filter_by(application_id="APP-001").first()
        assert row.fte_count is None
        assert row.numeric_field_notes is not None
        assert "fte_count" in row.numeric_field_notes
    finally:
        session.close()
