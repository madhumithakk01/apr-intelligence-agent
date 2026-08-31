"""Report assembly -- SPEC.md sections 5, 13.

Deterministic. Takes the finished run state and produces one structured
``report`` dict: a portfolio-level summary, one entry per scored
application, and a run-integrity section that surfaces (never hides)
ingestion collisions, failed branches, and which human gates fired.

This is the assembly half of the report stage; app.reporting.report_
renderer turns the dict this returns into Markdown. They live in one
package and share one shape so there is a single rendering path -- the
consolidation SPEC.md section 5 calls for. The pre-existing
app/services/report_{service,renderer}.py belong to the legacy
single-record API/CLI path and are removed once that path is migrated
off the legacy scoring services (not this branch's scope).

Nothing here calls an LLM or touches the filesystem: it is pure data
transformation over GraphState, safe to run inside a checkpointed node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.orchestration.shadow import normalize_mode
from app.orchestration.state import GraphState
from app.scoring import governance_params as gp

_TIME_DECISION_ORDER = ("Invest", "Migrate", "Tolerate", "Eliminate")


def _unevaluated_delivery(run_mode: str) -> Dict[str, Any]:
    """Fail-safe stamp when build_report is called without a delivery
    verdict (render_report always supplies one; this covers direct
    callers and tests). Never client-deliverable -- SPEC.md section 2."""
    return {
        "run_mode": run_mode,
        "client_deliverable": False,
        "shadow_signoff_on_record": False,
        "reason": "Delivery gate not evaluated for this report -- treated as not client-deliverable "
                  "(SPEC.md section 2).",
    }


def _sorted_application_ids(kernel_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Highest TIM-E score first; an unscored (None) row sorts last;
    ties broken by application id so the order is fully deterministic."""

    def key(application_id: str):
        score = kernel_results[application_id].get("tim_e_score")
        return (0, -score, application_id) if isinstance(score, (int, float)) else (1, 0, application_id)

    return sorted(kernel_results, key=key)


def _redundancy_for(application_id: str, verdicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = []
    for verdict in verdicts:
        a, b = verdict.get("application_id_a"), verdict.get("application_id_b")
        if application_id not in (a, b):
            continue
        recommendation = verdict.get("recommendation") or {}
        entries.append(
            {
                "counterpart": b if application_id == a else a,
                "typology": verdict.get("typology"),
                "recommendation": recommendation.get("recommendation") if isinstance(recommendation, dict) else None,
                "consolidation_blocked_by": (
                    recommendation.get("consolidation_blocked_by") if isinstance(recommendation, dict) else None
                ),
                "mandatory_review": bool(
                    verdict.get("mandatory_review")
                    or (isinstance(recommendation, dict) and recommendation.get("mandatory_review"))
                ),
                "rationale": recommendation.get("rationale") if isinstance(recommendation, dict) else verdict.get("rationale"),
            }
        )
    return entries


def _market_for(application_id: str, grounded_claims: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    findings = [f for f in grounded_claims.values() if f.get("application_id") == application_id]
    if not findings:
        return None
    products: List[Dict[str, Any]] = []
    seen = set()
    for finding in findings:
        for product in finding.get("products") or []:
            name = (product.get("name") or "").strip()
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            products.append(
                {
                    "name": name,
                    "vendor": (product.get("vendor") or "").strip(),
                    "claims": [
                        {"claim": c.get("claim"), "quote": c.get("quote"), "source_url": c.get("source_url")}
                        for c in (product.get("claims") or [])
                    ],
                }
            )
    no_viable = bool(findings) and all(f.get("no_viable_alternative_found") for f in findings) and not products
    return {
        "segments": [f.get("segment_id") for f in findings],
        "framings": [f.get("framing") for f in findings],
        "product_count": len(products),
        "products": products,
        "no_viable_alternative_found": no_viable,
    }


def _application_flags(
    application_id: str,
    narrative: Optional[Dict[str, Any]],
    redundancy: List[Dict[str, Any]],
    cost_outlier: Optional[Dict[str, Any]],
    phase2: List[Dict[str, Any]],
) -> List[str]:
    flags: List[str] = []
    if narrative is None:
        flags.append("No narrative was generated for this application.")
    elif narrative.get("source") == "structured_fallback":
        flags.append("Narrative failed grounding and shipped as structured bullets -- pending gate 5 review.")
    if any(entry.get("mandatory_review") for entry in redundancy):
        flags.append("A redundancy verdict for this application is pending human review (gate 3).")
    if cost_outlier and (cost_outlier.get("explainability") or {}).get("needs_review"):
        flags.append("A cost-outlier explainability check returned low confidence -- pending gate 4 review.")
    if phase2:
        labels = ", ".join(sorted({item.get("field_label") or item.get("field") or "" for item in phase2} - {""}))
        flags.append(f"Withheld in Phase 1, carried to Phase 2 discovery: {labels}.")
    return flags


def _portfolio_summary(
    kernel_results: Dict[str, Dict[str, Any]],
    verdicts: List[Dict[str, Any]],
    cost_outliers: List[Dict[str, Any]],
    grounded_claims: Dict[str, Dict[str, Any]],
    phase2_agenda: List[Dict[str, Any]],
) -> Dict[str, Any]:
    time_decisions: Dict[str, int] = {}
    cots_replace_candidates = 0
    for result in kernel_results.values():
        decision = result.get("tim_e_decision") or "Unscored"
        time_decisions[decision] = time_decisions.get(decision, 0) + 1
        if result.get("cots_meets_threshold"):
            cots_replace_candidates += 1

    redundancy_typologies: Dict[str, int] = {}
    for verdict in verdicts:
        typology = verdict.get("typology") or "Unclassified"
        redundancy_typologies[typology] = redundancy_typologies.get(typology, 0) + 1

    no_viable = sum(1 for f in grounded_claims.values() if f.get("no_viable_alternative_found"))

    return {
        "application_count": len(kernel_results),
        "time_decisions": time_decisions,
        "cots_replace_candidates": cots_replace_candidates,
        "cots_replace_threshold": gp.COTS_REPLACE_THRESHOLD,
        "redundancy_typologies": redundancy_typologies,
        "cost_outliers_flagged": len(cost_outliers),
        "market_segments_researched": len(grounded_claims),
        "no_viable_alternative_segments": no_viable,
        "phase2_discovery_items": len(phase2_agenda),
    }


def build_report(state: GraphState, *, delivery: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the structured report dict from a finished run. Always
    returns the full shape -- an empty portfolio yields zero-count
    summaries and an empty application list, not a missing section.

    ``delivery`` is the shadow-mode verdict
    (app.orchestration.shadow.ShadowLedger.delivery_status) the
    render_report node computes; when omitted the report is stamped
    not-client-deliverable (SPEC.md section 2)."""
    kernel_results = state.get("kernel_results") or {}
    narratives = state.get("narratives") or {}
    verdicts = state.get("verdicts") or []
    cost_outliers = state.get("cost_outliers") or []
    grounded_claims = state.get("grounded_claims") or {}
    phase2_agenda = state.get("phase2_agenda") or []
    applications_by_id = {
        a.get("application_id"): a for a in (state.get("applications") or []) if a.get("application_id")
    }

    cost_by_app = {c["application_id"]: c for c in cost_outliers if c.get("application_id")}
    phase2_by_app: Dict[str, List[Dict[str, Any]]] = {}
    for item in phase2_agenda:
        if item.get("application_id"):
            phase2_by_app.setdefault(item["application_id"], []).append(item)

    applications: List[Dict[str, Any]] = []
    for application_id in _sorted_application_ids(kernel_results):
        kernel_result = kernel_results[application_id]
        narrative = narratives.get(application_id)
        redundancy = _redundancy_for(application_id, verdicts)
        cost_outlier = cost_by_app.get(application_id)
        phase2 = phase2_by_app.get(application_id, [])
        source_row = applications_by_id.get(application_id, {})

        applications.append(
            {
                "application_id": application_id,
                "application_name": (
                    source_row.get("application_name")
                    or (narrative or {}).get("application_name")
                    or ""
                ),
                "narrative": (
                    {
                        "summary": narrative.get("summary"),
                        "source": narrative.get("source"),
                        "attempts": narrative.get("attempts"),
                    }
                    if narrative
                    else None
                ),
                "time_analysis": {
                    "score": kernel_result.get("tim_e_score"),
                    "decision": kernel_result.get("tim_e_decision"),
                    "raw_decision": kernel_result.get("tim_e_raw_decision"),
                    "floor_applied": kernel_result.get("floor_applied"),
                },
                "cots_analysis": {
                    "score": kernel_result.get("cots_score"),
                    "recommendation": kernel_result.get("cots_recommendation"),
                    "meets_threshold": kernel_result.get("cots_meets_threshold"),
                },
                "modernization_recommendation": kernel_result.get("modernization_recommendation"),
                "data_classification": kernel_result.get("security_classification"),
                "redundancy": redundancy,
                "cost_outlier": (
                    {
                        "direction": cost_outlier.get("direction"),
                        "cluster_id": cost_outlier.get("cluster_id"),
                        "cost_per_fte": cost_outlier.get("cost_per_fte"),
                        "explainability": cost_outlier.get("explainability"),
                    }
                    if cost_outlier
                    else None
                ),
                "market_alternatives": _market_for(application_id, grounded_claims),
                "phase2_discovery": [
                    {"field_label": item.get("field_label") or item.get("field"), "category": item.get("category")}
                    for item in phase2
                ],
                "flags": _application_flags(application_id, narrative, redundancy, cost_outlier, phase2),
            }
        )

    run_mode = normalize_mode(state.get("run_mode"))
    return {
        "run_id": state.get("run_id"),
        "data_sensitivity": state.get("data_sensitivity"),
        "run_mode": run_mode,
        "delivery": delivery or _unevaluated_delivery(run_mode),
        "portfolio_summary": _portfolio_summary(
            kernel_results, verdicts, cost_outliers, grounded_claims, phase2_agenda
        ),
        "applications": applications,
        "run_integrity": {
            "ingestion_collisions": state.get("ingestion_collisions") or [],
            "branch_failures": state.get("branch_failures") or [],
            "gate_decisions": state.get("gate_decisions") or {},
        },
    }
