"""HTTP layer for the bid-outcome data purge -- CLAUDE.md section 2.

  POST /api/retention/purge          clear every client-data store
  GET  /api/retention/purge/history  recent purge audit records

The purge is irreversible, so a real run requires an explicit
``confirm: true``; ``dry_run: true`` previews without deleting. All the
work lives in app.retention.purge.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.retention.purge import PurgePaths, purge_all_client_data, read_audit

router = APIRouter(prefix="/api/retention", tags=["retention"])


class PurgeRequest(BaseModel):
    reason: str = Field(min_length=1)
    """The bid outcome that authorizes the purge. Recorded in the audit log."""
    confirm: bool = False
    """Required for a real purge -- it is irreversible."""
    dry_run: bool = False
    """Report what would be removed; delete nothing."""


def _paths_dep() -> PurgePaths:
    return PurgePaths.defaults()


@router.post("/purge")
def purge(body: PurgeRequest, paths: PurgePaths = Depends(_paths_dep)) -> Dict[str, Any]:
    if not body.dry_run and not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="a real purge is irreversible -- pass confirm=true, or dry_run=true to preview",
        )
    report = purge_all_client_data(body.reason, dry_run=body.dry_run, paths=paths)
    return report.as_dict()


@router.get("/purge/history")
def purge_history(
    limit: int = Query(default=20, ge=1, le=200),
    paths: PurgePaths = Depends(_paths_dep),
) -> Dict[str, List[Dict[str, Any]]]:
    return {"purges": read_audit(limit, paths=paths)}
