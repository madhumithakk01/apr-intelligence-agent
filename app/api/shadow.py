"""HTTP layer for the shadow-mode delivery gate -- SPEC.md section 2.

  GET  /api/shadow          engagement delivery status: is client
                            delivery unlocked, which shadow runs have
                            completed, and every sign-off on record
  POST /api/shadow/signoff  record an internal reviewer's sign-off of a
                            completed shadow run -- an approved one is
                            what unlocks client-deliverable live runs

Thin: it delegates to app.orchestration.shadow.ShadowLedger and maps its
one error type to 404.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.orchestration.shadow import ShadowLedger, ShadowSignoffError, default_ledger

router = APIRouter(prefix="/api/shadow", tags=["shadow"])


class SignoffRequest(BaseModel):
    run_id: str
    reviewer: str
    decision: Any = "approved"
    """Verbatim reviewer decision. Only a recognized approval shape
    ("approved", or a dict with action/approve or signed_off: true)
    actually unlocks client delivery; anything else is recorded but
    leaves the engagement locked."""


def _ledger_dep() -> ShadowLedger:
    return default_ledger()


@router.get("")
def shadow_status(ledger: ShadowLedger = Depends(_ledger_dep)) -> Dict[str, Any]:
    return ledger.status()


@router.post("/signoff")
def sign_off_shadow_run(
    body: SignoffRequest, ledger: ShadowLedger = Depends(_ledger_dep)
) -> Dict[str, Any]:
    try:
        record = ledger.sign_off(body.run_id, reviewer=body.reviewer, decision=body.decision)
    except ShadowSignoffError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"signoff": record, "client_delivery_unlocked": ledger.is_client_delivery_unlocked()}
