"""Top-level run state for the orchestration graph -- CLAUDE.md section 13.

One state object flows through the whole pipeline (section 5). Every
field that a fanned-out branch can write carries an explicit reducer,
because LangGraph applies concurrent branch updates to the same key
without one only when it can -- and silently rejects them when it
can't. The reducers here are all key-disjoint merges or appends: a
branch writes only its own subject's key, never another branch's, so
two branches finishing in either order produce the same state. That is
what makes the parallel-isolation guarantee in CLAUDE.md section 8
("checkpointed independently so one branch's failure doesn't take down
the others") testable rather than aspirational.

This module defines shape only. Every stage that fills a field is a
pass-through stub on this branch -- see app/orchestration/nodes.py.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict


# --- reducers ---------------------------------------------------------------


def merge_by_key(left: Optional[dict], right: Optional[dict]) -> dict:
    """Shallow merge of two subject-keyed dicts.

    Safe under fan-out because each branch owns exactly one key (its own
    application / cluster / segment id). Last write wins on collision,
    which can only happen on a resumed re-execution of the same branch --
    where re-writing the same key with the same subject's result is the
    intended behavior.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def extend(left: Optional[list], right: Optional[list]) -> list:
    """Append-only accumulation. Order across parallel branches is not
    guaranteed and nothing downstream may depend on it -- consumers key
    off the records' own subject ids."""
    return list(left or []) + list(right or [])


# --- record shapes ----------------------------------------------------------


class ReviewItem(TypedDict, total=False):
    """One item queued for an internal human reviewer (CLAUDE.md section
    10). All review is internal to our own firm -- nothing here routes
    back to the client (section 2)."""

    gate: str
    subject_id: str
    reason: str
    payload: Dict[str, Any]


class BranchFailure(TypedDict, total=False):
    """A single fanned-out branch that failed. Recorded, never raised
    past the branch boundary: one segment's search failure must not take
    the other 99 rows' run down with it (CLAUDE.md section 8)."""

    branch_kind: str  # "disclosure" | "qualitative" | "redundancy" | "market"
    subject_id: str
    error: str


class StageStatus(TypedDict, total=False):
    """Marker written by every stub node so a run over synthetic
    fixtures still proves the topology executed end to end."""

    stage: str
    status: str  # "stub" once implemented stages exist, this becomes "complete"
    implemented_by_branch: str


# --- run state --------------------------------------------------------------


class GraphState(TypedDict, total=False):
    run_id: str

    data_sensitivity: str
    """"real" | "synthetic" -- CLAUDE.md section 11's routing flag, carried
    on the run itself so every LLM-backed stage inherits one declared
    value instead of deciding locally. Passed to llm.providers.get_completion
    by every stage that calls it, starting with disclosure classification
    (branch 6)."""

    dataset_path: Optional[str]
    """Excel workbook to load if `applications` isn't already populated
    (app.orchestration.nodes.ingest). Omitted (or `applications` supplied
    directly) is how tests and any future caller that already has rows in
    hand bypass the file entirely."""

    applications: List[Dict[str, Any]]
    """Ingested rows, one per application, ApplicationInput-shaped
    (app.ingestion.row_mapping). Loaded from `dataset_path` via
    app.ingestion.excel_loader.ExcelLoader when not supplied directly."""

    ingestion_collisions: List[Dict[str, Any]]
    """Application IDs that occurred more than once in the loaded
    workbook. Surfaced, never silently dropped (CLAUDE.md section 4 bug
    7) -- every colliding row is excluded from `applications` rather than
    picking a first-write winner."""

    disclosure: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    """application_id -> {"results": {field: DisclosureResult-as-dict},
    "gated_application": application dict with every non-Answered field
    nulled out} (section 6). Gates every downstream scoring step for the
    field it classifies."""

    phase2_agenda: Annotated[List[Dict[str, Any]], extend]
    """One item per non-Answered classified field, across every
    application -- the disclosure classifier's output doubling as the
    Phase 2 discovery/interview agenda (section 6)."""

    rubrics: Dict[str, Dict[str, Any]]
    rubric_signoff: Dict[str, Any]

    qualitative_scores: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    kernel_results: Annotated[Dict[str, Dict[str, Any]], merge_by_key]

    clusters: List[Dict[str, Any]]
    profiles: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    verdicts: Annotated[List[Dict[str, Any]], extend]
    recommendations: List[Dict[str, Any]]

    cost_outliers: List[Dict[str, Any]]

    segments: List[Dict[str, Any]]
    """Redundancy-surviving segments -- the Market Intelligence fan-out
    unit (CLAUDE.md section 8: once per segment, not once per raw app, so
    a Scale-Tiered Overlap yields two differently-framed targets)."""

    market_findings: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    grounded_claims: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    narratives: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    report: Dict[str, Any]

    review_queue: Annotated[List[ReviewItem], extend]
    gate_decisions: Annotated[Dict[str, Dict[str, Any]], merge_by_key]
    branch_failures: Annotated[List[BranchFailure], extend]
    stage_log: Annotated[List[StageStatus], extend]


# --- fan-out task payloads --------------------------------------------------
# Send() payloads. Each carries only what its branch needs plus the run's
# sensitivity flag -- a branch never receives the whole portfolio, so one
# branch cannot read or corrupt another's inputs.


class RowTask(TypedDict, total=False):
    run_id: str
    data_sensitivity: str
    application: Dict[str, Any]
    rubrics: Dict[str, Dict[str, Any]]


class ClusterTask(TypedDict, total=False):
    run_id: str
    data_sensitivity: str
    cluster: Dict[str, Any]
    profiles: Dict[str, Dict[str, Any]]


class SegmentTask(TypedDict, total=False):
    run_id: str
    data_sensitivity: str
    segment: Dict[str, Any]
