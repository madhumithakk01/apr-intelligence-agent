"""Report rendering -- CLAUDE.md sections 5, 13.

Deterministic. One function, one output format: the structured report
dict from app.reporting.report_service becomes a Markdown string. This
is the single rendering path CLAUDE.md section 5 asks for -- no second
renderer, no divergent section templates.

Markdown only. PDF generation and file writing are a delivery concern
for whatever serves the report (the async job endpoint, branch 16), not
something a checkpointed pipeline node should do as a side effect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_NA = "_not recorded_"


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return _NA
    return str(value)


def _score(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return _NA
    return f"{value:g}/100"


def _counts_line(counts: Dict[str, int], order: Optional[List[str]] = None) -> str:
    if not counts:
        return _NA
    keys = [k for k in (order or []) if k in counts] + [k for k in counts if not order or k not in order]
    return ", ".join(f"{key}: {counts[key]}" for key in keys)


def _portfolio_section(report: Dict[str, Any]) -> List[str]:
    summary = report.get("portfolio_summary") or {}
    lines = [
        "## Portfolio Summary",
        "",
        f"- Applications scored: **{summary.get('application_count', 0)}**",
        f"- TIM-E decisions: {_counts_line(summary.get('time_decisions') or {}, ['Invest', 'Migrate', 'Tolerate', 'Eliminate'])}",
        f"- COTS-replace candidates (fit score >= {summary.get('cots_replace_threshold', _NA)}): "
        f"**{summary.get('cots_replace_candidates', 0)}**",
        f"- Redundancy verdicts: {_counts_line(summary.get('redundancy_typologies') or {})}",
        f"- Cost outliers flagged: **{summary.get('cost_outliers_flagged', 0)}**",
        f"- Market segments researched: **{summary.get('market_segments_researched', 0)}** "
        f"({summary.get('no_viable_alternative_segments', 0)} with no viable COTS alternative)",
        f"- Phase 2 discovery items: **{summary.get('phase2_discovery_items', 0)}**",
        "",
    ]
    return lines


def _application_section(entry: Dict[str, Any]) -> List[str]:
    name = _fmt(entry.get("application_name"))
    lines = [f"### {name} ({_fmt(entry.get('application_id'))})", ""]

    narrative = entry.get("narrative")
    if narrative and narrative.get("summary"):
        lines += [narrative["summary"].strip(), ""]
        if narrative.get("source") == "structured_fallback":
            lines += ["> Generated narrative failed grounding; the bullets above are the deterministic fallback.", ""]
    else:
        lines += [f"{_NA} (no narrative generated)", ""]

    time_analysis = entry.get("time_analysis") or {}
    cots = entry.get("cots_analysis") or {}
    lines.append(f"- **TIM-E:** {_score(time_analysis.get('score'))} -- {_fmt(time_analysis.get('decision'))}")
    if time_analysis.get("floor_applied"):
        lines.append(f"  - Non-compensatory floor applied: {time_analysis['floor_applied']}")
    lines.append(
        f"- **COTS fit:** {_score(cots.get('score'))} -- {_fmt(cots.get('recommendation'))}"
        + (" (meets replace threshold)" if cots.get("meets_threshold") else "")
    )
    lines.append(f"- **Modernization:** {_fmt(entry.get('modernization_recommendation'))}")
    lines.append(f"- **Data classification:** {_fmt(entry.get('data_classification'))}")

    for red in entry.get("redundancy") or []:
        blocked = f" (consolidation blocked by {red['consolidation_blocked_by']})" if red.get("consolidation_blocked_by") else ""
        lines.append(
            f"- **Redundancy vs {_fmt(red.get('counterpart'))}:** {_fmt(red.get('typology'))} -- "
            f"{_fmt(red.get('recommendation'))}{blocked}"
        )

    outlier = entry.get("cost_outlier")
    if outlier:
        exp = outlier.get("explainability") or {}
        verdict = (
            "explainable" if exp.get("explainable") is True
            else "not explainable" if exp.get("explainable") is False
            else "not assessed"
        )
        lines.append(
            f"- **Cost outlier:** {_fmt(outlier.get('direction'))} cost-per-FTE in cluster "
            f"{_fmt(outlier.get('cluster_id'))} -- {verdict}"
        )

    market = entry.get("market_alternatives")
    if market:
        if market.get("no_viable_alternative_found"):
            lines.append("- **Market alternatives:** no viable COTS alternative found.")
        else:
            lines.append(f"- **Market alternatives:** {market.get('product_count', 0)} grounded")
            for product in market.get("products") or []:
                vendor = f" ({product['vendor']})" if product.get("vendor") else ""
                lines.append(f"  - {_fmt(product.get('name'))}{vendor}")
                for claim in product.get("claims") or []:
                    lines.append(f"    - {_fmt(claim.get('claim'))} -- source: {_fmt(claim.get('source_url'))}")

    phase2 = entry.get("phase2_discovery") or []
    if phase2:
        labels = ", ".join(_fmt(item.get("field_label")) for item in phase2)
        lines.append(f"- **Phase 2 discovery:** {labels}")

    for flag in entry.get("flags") or []:
        lines.append(f"- :warning: {flag}")

    lines.append("")
    return lines


def _run_integrity_section(report: Dict[str, Any]) -> List[str]:
    integrity = report.get("run_integrity") or {}
    collisions = integrity.get("ingestion_collisions") or []
    failures = integrity.get("branch_failures") or []
    gate_decisions = integrity.get("gate_decisions") or {}

    lines = ["## Run Integrity", ""]
    if not (collisions or failures or gate_decisions):
        lines += ["- Clean run: no ingestion collisions, no failed branches, no gates fired.", ""]
        return lines

    if collisions:
        lines.append(f"- **Ingestion collisions:** {len(collisions)} Application ID(s) appeared more than once and were excluded:")
        for collision in collisions:
            lines.append(f"  - {_fmt(collision.get('application_id'))} ({collision.get('occurrences')} occurrences)")
    if failures:
        lines.append(f"- **Failed branches:** {len(failures)} (recorded, did not stop the run):")
        for failure in failures:
            lines.append(
                f"  - {_fmt(failure.get('branch_kind'))} / {_fmt(failure.get('subject_id'))}: {_fmt(failure.get('error'))}"
            )
    if gate_decisions:
        lines.append("- **Human gates fired:**")
        for gate, decision in gate_decisions.items():
            reviewed = decision.get("reviewed_subject_ids") or []
            suffix = f" -- reviewed: {', '.join(reviewed)}" if reviewed else ""
            lines.append(f"  - {gate}: {_fmt(decision.get('decision'))}{suffix}")
    lines.append("")
    return lines


def _delivery_banner(report: Dict[str, Any]) -> List[str]:
    """A report that is not client-deliverable says so, loudly, right
    under the title -- CLAUDE.md section 2."""
    delivery = report.get("delivery") or {}
    if delivery.get("client_deliverable"):
        return []
    label = (
        "SHADOW RUN -- INTERNAL REVIEW ONLY"
        if delivery.get("run_mode") == "shadow"
        else "NOT CLIENT-DELIVERABLE"
    )
    return [
        f"> **{label}.** {_fmt(delivery.get('reason'))}",
        "",
    ]


def render_markdown(report: Dict[str, Any]) -> str:
    """The one rendering path: structured report dict -> Markdown."""
    lines = ["# APR Portfolio Rationalization Report", ""]
    lines += _delivery_banner(report)
    lines += [
        f"- Run: {_fmt(report.get('run_id'))}",
        f"- Mode: {_fmt(report.get('run_mode'))}",
        f"- Data sensitivity: {_fmt(report.get('data_sensitivity'))}",
        "",
    ]
    lines += _portfolio_section(report)

    lines += ["## Applications", ""]
    applications = report.get("applications") or []
    if applications:
        for entry in applications:
            lines += _application_section(entry)
    else:
        lines += ["_No applications were scored in this run._", ""]

    lines += _run_integrity_section(report)
    return "\n".join(lines).rstrip() + "\n"
