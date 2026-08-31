"""Report rendering -- SPEC.md sections 5, 13.

Deterministic string output; no mocks.
"""

from __future__ import annotations

from app.reporting import report_renderer, report_service
from test_report_service import _state


def _md(**overrides):
    return report_renderer.render_markdown(report_service.build_report(_state(**overrides)))


def test_the_report_has_the_three_top_level_sections():
    md = _md()
    assert md.startswith("# APR Portfolio Rationalization Report")
    assert "## Portfolio Summary" in md
    assert "## Applications" in md
    assert "## Run Integrity" in md


def test_each_scored_application_gets_its_own_heading_in_score_order():
    md = _md()
    assert md.index("### Claims Intake (APP-1)") < md.index("### Claims Intake Lite (APP-2)") < md.index("### Ledger (APP-3)")


def test_the_generated_narrative_is_rendered_verbatim():
    md = _md()
    assert "Claims Intake is a Migrate at 74/100." in md


def test_a_structured_fallback_narrative_carries_a_visible_note():
    md = _md()
    assert "the bullets above are the deterministic fallback" in md


def test_scores_render_out_of_100_and_missing_values_show_a_placeholder():
    md = _md()
    assert "74/100" in md
    assert report_renderer._NA in md  # APP-3 has no score / classification


def test_no_viable_alternative_is_rendered_as_a_sentence_not_an_empty_list():
    md = _md()
    ledger = md[md.index("### Ledger (APP-3)"):]
    assert "no viable COTS alternative found" in ledger


def test_flags_are_rendered_with_a_warning_marker():
    md = _md()
    assert ":warning:" in md
    assert "pending gate 5 review" in md


def test_a_clean_run_integrity_section_says_so():
    md = _md(gate_decisions={})
    assert "Clean run: no ingestion collisions" in md


def test_collisions_and_failed_branches_are_listed_when_present():
    md = _md(
        ingestion_collisions=[{"application_id": "APP-9", "occurrences": 3}],
        branch_failures=[{"branch_kind": "market", "subject_id": "SEG-X", "error": "RuntimeError: boom"}],
    )
    assert "APP-9 (3 occurrences)" in md
    assert "market / SEG-X: RuntimeError: boom" in md
    assert "Clean run" not in md


def test_an_empty_run_renders_without_error():
    md = report_renderer.render_markdown(report_service.build_report({"run_id": "run-empty"}))
    assert "_No applications were scored in this run._" in md
    assert md.endswith("\n")


def test_rendering_is_deterministic():
    assert _md() == _md()
