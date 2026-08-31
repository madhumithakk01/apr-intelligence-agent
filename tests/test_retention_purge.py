"""Bid-outcome data purge -- SPEC.md section 2.

One entry point clears every client-data store on the bid-outcome
trigger. Tests run against a PurgePaths rooted at tmp_path, so nothing
touches a real store.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import retention as retention_api
from app.retention import purge as pg


def _make_app_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE applications (id INTEGER PRIMARY KEY, application_id TEXT);
        CREATE TABLE market_products (id INTEGER PRIMARY KEY, product_name TEXT);
        CREATE TABLE analysis_runs (id INTEGER PRIMARY KEY, application_id TEXT);
        INSERT INTO applications (application_id) VALUES ('APP-1'), ('APP-2');
        INSERT INTO market_products (product_name) VALUES ('Guidewire');
        INSERT INTO analysis_runs (application_id) VALUES ('APP-1');
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def populated(tmp_path):
    """A PurgePaths where every store exists and holds something."""
    kb = tmp_path / "knowledge_db"
    reports = tmp_path / "reports"
    uploads = kb / "batch_uploads"
    uploads.mkdir(parents=True)
    reports.mkdir(parents=True)

    (kb / "orchestration_checkpoints.db").write_bytes(b"sqlite-ish")
    (kb / "orchestration_checkpoints.db-wal").write_bytes(b"wal")
    (kb / "market_intelligence_checkpoints.db").write_bytes(b"sqlite-ish")
    (kb / "shadow_ledger.json").write_text('{"shadow_runs": [], "signoffs": []}', encoding="utf-8")
    (uploads / "client.xlsx").write_bytes(b"workbook")
    (uploads / "client2.xlsx").write_bytes(b"workbook")
    (reports / "APP-1_report.md").write_text("# report", encoding="utf-8")
    _make_app_db(kb / "apr.db")

    return pg.PurgePaths(
        orchestration_checkpoints=kb / "orchestration_checkpoints.db",
        market_checkpoints=kb / "market_intelligence_checkpoints.db",
        app_db=kb / "apr.db",
        shadow_ledger=kb / "shadow_ledger.json",
        uploads_dir=uploads,
        reports_dir=reports,
        audit_log=kb / "purge_audit.jsonl",
    )


def _by_store(report):
    return {s["store"]: s for s in report.as_dict()["stores"]}


# --- the real purge --------------------------------------------------


def test_a_full_purge_clears_every_store(populated):
    report = pg.purge_all_client_data("bid awarded to vendor X", paths=populated)

    assert report.ok
    stores = _by_store(report)
    assert stores["orchestration_checkpoints"]["removed"] is True
    assert stores["market_checkpoints"]["removed"] is True
    assert stores["shadow_ledger"]["removed"] is True
    assert stores["uploaded_workbooks"]["detail"] == "2 file(s)"
    assert stores["generated_reports"]["removed"] is True

    assert not populated.orchestration_checkpoints.exists()
    assert not (populated.orchestration_checkpoints.parent / "orchestration_checkpoints.db-wal").exists()
    assert not populated.market_checkpoints.exists()
    assert not populated.shadow_ledger.exists()
    assert list(populated.uploads_dir.rglob("*")) == []
    assert list(populated.reports_dir.rglob("*")) == []


def test_the_app_db_keeps_its_schema_but_loses_every_client_row(populated):
    report = pg.purge_all_client_data("bid lost", paths=populated)

    assert _by_store(report)["app_db_rows"]["detail"].startswith("4 row(s) across")
    conn = sqlite3.connect(str(populated.app_db))
    try:
        for table in ("applications", "market_products", "analysis_runs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # rows gone
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"applications", "market_products", "analysis_runs"} <= names  # tables kept
    finally:
        conn.close()


def test_langsmith_is_reported_as_a_manual_step_not_performed(populated):
    stores = _by_store(pg.purge_all_client_data("bid concluded", paths=populated))
    assert stores["langsmith_runs"]["removed"] is False
    assert "LangSmith" in stores["langsmith_runs"]["detail"]


def test_a_purge_requires_a_reason(populated):
    with pytest.raises(ValueError):
        pg.purge_all_client_data("   ", paths=populated)


# --- dry run -------------------------------------------------------


def test_a_dry_run_reports_what_would_go_and_deletes_nothing(populated):
    report = pg.purge_all_client_data("preview", dry_run=True, paths=populated)

    assert report.dry_run is True
    assert _by_store(report)["uploaded_workbooks"]["removed"] is True  # would remove
    assert populated.orchestration_checkpoints.exists()
    assert populated.shadow_ledger.exists()
    assert len(list(populated.uploads_dir.iterdir())) == 2
    conn = sqlite3.connect(str(populated.app_db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 2  # untouched
    finally:
        conn.close()
    assert not populated.audit_log.exists()  # a preview is not audited


# --- resilience --------------------------------------------------


def test_missing_stores_are_reported_not_errored(tmp_path):
    kb = tmp_path / "knowledge_db"
    paths = pg.PurgePaths(
        orchestration_checkpoints=kb / "cp.db",
        market_checkpoints=kb / "mkt.db",
        app_db=kb / "apr.db",
        shadow_ledger=kb / "ledger.json",
        uploads_dir=kb / "uploads",
        reports_dir=tmp_path / "reports",
        audit_log=kb / "audit.jsonl",
    )
    report = pg.purge_all_client_data("nothing to do", paths=paths)

    assert report.ok
    assert all(s["error"] is None for s in report.as_dict()["stores"])
    assert {s["detail"] for s in report.as_dict()["stores"] if s["store"].endswith(("checkpoints", "shadow_ledger"))} == {
        "not present"
    }


def test_purge_is_idempotent(populated):
    pg.purge_all_client_data("first", paths=populated)
    second = pg.purge_all_client_data("second", paths=populated)

    assert second.ok
    stores = _by_store(second)
    assert stores["orchestration_checkpoints"]["detail"] == "not present"
    assert stores["shadow_ledger"]["detail"] == "not present"
    assert stores["app_db_rows"]["removed"] is False  # 0 rows the second time


def test_one_failing_store_does_not_stop_the_others(populated, monkeypatch):
    def boom(name, *a, **k):
        raise RuntimeError("disk error")

    monkeypatch.setattr(pg, "_empty_dir", boom)
    report = pg.purge_all_client_data("bid lost", paths=populated)

    assert report.ok is False
    stores = _by_store(report)
    assert stores["uploaded_workbooks"]["error"] == "RuntimeError: disk error"
    assert stores["orchestration_checkpoints"]["removed"] is True  # still ran
    assert not populated.shadow_ledger.exists()


# --- audit ------------------------------------------------------


def test_every_real_purge_appends_one_audit_line(populated):
    pg.purge_all_client_data("bid awarded to X", paths=populated)
    pg.purge_all_client_data("cleanup pass", paths=populated)

    records = pg.read_audit(paths=populated)
    assert [r["reason"] for r in records] == ["bid awarded to X", "cleanup pass"]
    assert all(r["dry_run"] is False for r in records)
    raw_lines = populated.audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw_lines) == 2 and json.loads(raw_lines[0])["ok"] is True


def test_read_audit_is_empty_when_nothing_has_been_purged(populated):
    assert pg.read_audit(paths=populated) == []


# --- HTTP surface --------------------------------------------


@pytest.fixture
def client(populated):
    app = FastAPI()
    app.include_router(retention_api.router)
    app.dependency_overrides[retention_api._paths_dep] = lambda: populated
    with TestClient(app) as c:
        yield c


def test_purge_endpoint_requires_confirm_for_a_real_run(client):
    resp = client.post("/api/retention/purge", json={"reason": "bid lost"})
    assert resp.status_code == 400
    assert "confirm" in resp.json()["detail"]


def test_purge_endpoint_runs_with_confirm(client, populated):
    resp = client.post("/api/retention/purge", json={"reason": "bid lost", "confirm": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["dry_run"] is False
    assert not populated.shadow_ledger.exists()


def test_purge_endpoint_dry_run_needs_no_confirm(client, populated):
    resp = client.post("/api/retention/purge", json={"reason": "preview", "dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
    assert populated.shadow_ledger.exists()


def test_purge_endpoint_rejects_an_empty_reason(client):
    assert client.post("/api/retention/purge", json={"reason": ""}).status_code == 422


def test_purge_history_endpoint_lists_audit_records(client):
    client.post("/api/retention/purge", json={"reason": "run one", "confirm": True})
    history = client.get("/api/retention/purge/history").json()["purges"]
    assert [r["reason"] for r in history] == ["run one"]
