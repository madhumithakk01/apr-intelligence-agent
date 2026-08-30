import pandas as pd
import pytest

from app.ingestion.excel_loader import ExcelLoader


def _write_workbook(path, sheet1_rows, sheet2_rows):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(sheet1_rows).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame(sheet2_rows).to_excel(writer, sheet_name="Sheet2", index=False)


@pytest.fixture
def colliding_workbook(tmp_path):
    path = tmp_path / "dataset.xlsx"
    _write_workbook(
        path,
        sheet1_rows=[{"Application ID": "APP-001", "Application Name": "SalesHub CRM"}],
        sheet2_rows=[
            {"Application ID": "APP-001", "Application Name": "Global ERP Core"},
            {"Application ID": "APP-002", "Application Name": "Payroll System"},
        ],
    )
    return path


@pytest.fixture
def clean_workbook(tmp_path):
    path = tmp_path / "dataset.xlsx"
    _write_workbook(
        path,
        sheet1_rows=[{"Application ID": "APP-001", "Application Name": "SalesHub CRM"}],
        sheet2_rows=[{"Application ID": "APP-002", "Application Name": "Payroll System"}],
    )
    return path


def test_load_never_drops_colliding_rows(colliding_workbook):
    df = ExcelLoader(str(colliding_workbook)).load()

    assert len(df) == 3
    assert (df["Application ID"] == "APP-001").sum() == 2


def test_find_duplicate_application_ids_reports_collision(colliding_workbook):
    df = ExcelLoader(str(colliding_workbook)).load()

    collisions = ExcelLoader.find_duplicate_application_ids(df)

    assert set(collisions.keys()) == {"APP-001"}
    occurrences = collisions["APP-001"]
    assert len(occurrences) == 2
    sheets = {o["source_sheet"] for o in occurrences}
    assert sheets == {"Sheet1", "Sheet2"}
    names = {o["application_name"] for o in occurrences}
    assert names == {"SalesHub CRM", "Global ERP Core"}
    assert all(o["source_row"] is not None for o in occurrences)


def test_find_duplicate_application_ids_empty_when_no_collisions(clean_workbook):
    df = ExcelLoader(str(clean_workbook)).load()

    assert ExcelLoader.find_duplicate_application_ids(df) == {}
