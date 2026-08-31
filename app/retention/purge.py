"""Bid-outcome data purge -- CLAUDE.md section 2.

Cached and stored client data has one deletion trigger -- the bid
concluded -- not a TTL, and nothing here assumes indefinite retention.
This module is that trigger's single entry point:
``purge_all_client_data(reason)`` clears every store at once and writes
an audit line, because "the bid is over" authorizes deleting everything
together rather than expiring rows by age.

Stores cleared:
  * the LangGraph checkpoint stores (orchestration + the market
    subgraph), including their -wal/-shm sidecars;
  * the application DB rows -- ``applications`` / ``market_products`` /
    ``analysis_runs`` -- leaving the schema in place;
  * the shadow-mode engagement ledger;
  * uploaded client workbooks (``knowledge_db/batch_uploads/``);
  * generated reports (``reports/``).

LangSmith runs live in a third-party project and are deleted there, out
of band -- this module reports that as a manual step it did not perform.

``dry_run=True`` reports what would go and touches nothing. Every store
is handled independently: one failing does not stop the others, and its
error is carried in the report.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from app.orchestration.checkpointer import purge_checkpoint_store

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# applications is the live table; market_products and analysis_runs were
# dropped with the legacy path but a pre-existing deployment may still
# carry them with client data, so the purge clears them if present.
_APP_DB_CLIENT_TABLES = ("applications", "market_products", "analysis_runs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PurgePaths:
    orchestration_checkpoints: Path
    market_checkpoints: Path
    app_db: Path
    shadow_ledger: Path
    uploads_dir: Path
    reports_dir: Path
    audit_log: Path

    @classmethod
    def defaults(cls) -> "PurgePaths":
        kb = _PROJECT_ROOT / "knowledge_db"
        return cls(
            orchestration_checkpoints=kb / "orchestration_checkpoints.db",
            market_checkpoints=kb / "market_intelligence_checkpoints.db",
            app_db=kb / "apr.db",
            shadow_ledger=Path(os.getenv("SHADOW_LEDGER_PATH") or (kb / "shadow_ledger.json")),
            uploads_dir=kb / "batch_uploads",
            reports_dir=_PROJECT_ROOT / "reports",
            audit_log=kb / "purge_audit.jsonl",
        )


@dataclass
class StoreResult:
    store: str
    removed: bool
    detail: str
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PurgeReport:
    reason: str
    dry_run: bool
    started_at: str
    finished_at: str
    stores: List[StoreResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(store.error is None for store in self.stores)

    def as_dict(self) -> dict:
        return {
            "reason": self.reason,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "stores": [store.as_dict() for store in self.stores],
        }


# --- per-store handlers -------------------------------------------------


def _purge_checkpoint(name: str, path: Path, *, dry_run: bool) -> StoreResult:
    sidecars = [Path(str(path) + suffix) for suffix in ("-wal", "-shm")]
    present = [p for p in (path, *sidecars) if p.exists()]
    if not present:
        return StoreResult(name, removed=False, detail="not present")
    if not dry_run:
        purge_checkpoint_store(path)
        for sidecar in sidecars:
            if sidecar.exists():
                sidecar.unlink()
    return StoreResult(name, removed=True, detail=f"{len(present)} file(s)")


def _purge_app_db_rows(name: str, db_path: Path, *, dry_run: bool) -> StoreResult:
    if not db_path.exists():
        return StoreResult(name, removed=False, detail="not present")
    connection = sqlite3.connect(str(db_path))
    try:
        counts = {}
        for table in _APP_DB_CLIENT_TABLES:
            try:
                counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = None  # table not created yet
        present = {t: c for t, c in counts.items() if c is not None}
        if not present:
            return StoreResult(name, removed=False, detail="no client tables present")
        total = sum(present.values())
        if not dry_run and total:
            for table in present:
                connection.execute(f"DELETE FROM {table}")
            connection.commit()
            connection.execute("VACUUM")
        return StoreResult(
            name,
            removed=total > 0,
            detail=f"{total} row(s) across " + ", ".join(sorted(present)),
        )
    finally:
        connection.close()


def _delete_file(name: str, path: Path, *, dry_run: bool) -> StoreResult:
    if not path.exists():
        return StoreResult(name, removed=False, detail="not present")
    if not dry_run:
        path.unlink()
    return StoreResult(name, removed=True, detail="1 file")


def _empty_dir(name: str, directory: Path, *, dry_run: bool) -> StoreResult:
    if not directory.exists():
        return StoreResult(name, removed=False, detail="not present")
    files = [p for p in directory.rglob("*") if p.is_file()]
    if not files:
        return StoreResult(name, removed=False, detail="empty")
    if not dry_run:
        for f in files:
            f.unlink()
        for sub in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
            try:
                sub.rmdir()
            except OSError:
                pass
    return StoreResult(name, removed=True, detail=f"{len(files)} file(s)")


def _langsmith_note(name: str) -> StoreResult:
    project = os.getenv("LANGSMITH_PROJECT") or "apr-intelligence-agent"
    return StoreResult(
        name,
        removed=False,
        detail=f"not performed here -- delete the '{project}' LangSmith project in LangSmith",
    )


def _run(handler: Callable[..., StoreResult], name: str, *args, **kwargs) -> StoreResult:
    try:
        return handler(name, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- one store's failure must not stop the rest
        return StoreResult(name, removed=False, detail="failed", error=f"{type(exc).__name__}: {exc}")


# --- audit ----------------------------------------------------------


def _append_audit(audit_log: Path, report: PurgeReport) -> None:
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report.as_dict(), sort_keys=True) + "\n")


def read_audit(limit: int = 20, *, paths: Optional[PurgePaths] = None) -> List[dict]:
    """The most recent purge records, newest last. Empty if none."""
    audit_log = (paths or PurgePaths.defaults()).audit_log
    try:
        lines = audit_log.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines[-max(limit, 0):]:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# --- entry point --------------------------------------------------


def purge_all_client_data(
    reason: str, *, dry_run: bool = False, paths: Optional[PurgePaths] = None
) -> PurgeReport:
    """Clear every client-data store. ``reason`` (the bid outcome) is
    required and recorded. Never raises for a store failure -- check
    ``report.ok`` and the per-store ``error`` fields."""
    if not (reason or "").strip():
        raise ValueError("a purge requires a non-empty reason -- the bid outcome that authorizes it")
    paths = paths or PurgePaths.defaults()

    started = _now()
    stores = [
        _run(_purge_checkpoint, "orchestration_checkpoints", paths.orchestration_checkpoints, dry_run=dry_run),
        _run(_purge_checkpoint, "market_checkpoints", paths.market_checkpoints, dry_run=dry_run),
        _run(_purge_app_db_rows, "app_db_rows", paths.app_db, dry_run=dry_run),
        _run(_delete_file, "shadow_ledger", paths.shadow_ledger, dry_run=dry_run),
        _run(_empty_dir, "uploaded_workbooks", paths.uploads_dir, dry_run=dry_run),
        _run(_empty_dir, "generated_reports", paths.reports_dir, dry_run=dry_run),
        _langsmith_note("langsmith_runs"),
    ]
    report = PurgeReport(
        reason=reason.strip(), dry_run=dry_run, started_at=started, finished_at=_now(), stores=stores
    )
    if not dry_run:
        try:
            _append_audit(paths.audit_log, report)
            report.stores.append(StoreResult("audit", removed=True, detail="recorded"))
        except Exception as exc:  # noqa: BLE001
            report.stores.append(
                StoreResult("audit", removed=False, detail="failed", error=f"{type(exc).__name__}: {exc}")
            )
    return report


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Purge all client data on the bid-outcome trigger (CLAUDE.md section 2).")
    parser.add_argument("--reason", required=True, help="the bid outcome authorizing the purge")
    parser.add_argument("--dry-run", action="store_true", help="report what would be removed; delete nothing")
    parser.add_argument("--confirm", action="store_true", help="required for a real (non-dry-run) purge")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.confirm:
        parser.error("a real purge is irreversible -- pass --confirm (or use --dry-run)")

    report = purge_all_client_data(args.reason, dry_run=args.dry_run)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
