"""Report assembly -- SPEC.md sections 5, 13.

Pure data transformation over GraphState; no mocks needed anywhere.
"""

from __future__ import annotations

from app.reporting import report_service
from app.scoring import governance_params as gp


def _state(**overrides):
    base = {
        "run_id": "run-1",
        "data_sensitivity": "synthetic",
        "applications": [
            {"application_id": "APP-1", "application_name": "Claims Intake"},
            {"application_id": "APP-2", "application_name": "Claims Intake Lite"},
            {"application_id": "APP-3", "application_name": "Ledger"},
        ],
        "kernel_results": {
            "APP-1": {"tim_e_score": 74, "tim_e_decision": "Migrate", "tim_e_raw_decision": "Migrate",
                      "floor_applied": None, "security_classification": "Confidential",
                      "cots_score": 68, "cots_recommendation": "Replace with COTS", "cots_meets_threshold": True,
                      "modernization_recommendation": "Refactor then replace"},
            "APP-2": {"tim_e_score": 41, "tim_e_decision": "Eliminate", "tim_e_raw_decision": "Eliminate",
                      "floor_applied": None, "security_classification": "Internal use only",
                      "cots_score": 55, "cots_recommendation": "Retain custom build", "cots_meets_threshold": False,
                      "modernization_recommendation": "Retire after consolidation"},
            "APP-3": {"tim_e_score": None, "tim_e_decision": "Insufficient Data", "tim_e_raw_decision": None,
                      "floor_applied": None, "security_classification": None,
                      "cots_score": None, "cots_recommendation": "Not assessed", "cots_meets_threshold": False,
                      "modernization_recommendation": "Deferred"},
        },
        "narratives": {
            "APP-1": {"application_id": "APP-1", "application_name": "Claims Intake",
                      "summary": "Claims Intake is a Migrate at 74/100.", "source": "generated", "attempts": 1},
            "APP-2": {"application_id": "APP-2", "application_name": "Claims Intake Lite",
                      "summary": "- Decision: Eliminate (TIM-E score 41/100).", "source": "structured_fallback",
                      "attempts": 2},
        },
        "verdicts": [
            {"application_id_a": "APP-1", "application_id_b": "APP-2", "typology": "Scale-Tiered Overlap",
             "mandatory_review": False,
             "recommendation": {"recommendation": "Migrate the light tier onto the heavy platform",
                                "consolidation_blocked_by": None, "mandatory_review": True,
                                "rationale": "normalized cost supports it"}},
        ],
        "cost_outliers": [
            {"application_id": "APP-1", "cluster_id": "CL-CLAIMS", "direction": "high", "cost_per_fte": 45000.0,
             "cluster_stats": {}, "explainability": {"explainable": False, "confidence": 0.4,
                                                     "rationale": "nothing in the profile explains it", "needs_review": True}},
        ],
        "grounded_claims": {
            "SEG-APP-1-tier_enterprise": {"segment_id": "SEG-APP-1-tier_enterprise", "application_id": "APP-1",
                                          "framing": "tier_enterprise", "no_viable_alternative_found": False,
                                          "products": [{"name": "Guidewire", "vendor": "Guidewire Inc",
                                                        "claims": [{"claim": "Cloud claims platform.", "quote": "cloud claims",
                                                                    "source_url": "https://g.example"}]}]},
            "SEG-APP-1-tier_light": {"segment_id": "SEG-APP-1-tier_light", "application_id": "APP-1",
                                     "framing": "tier_light", "no_viable_alternative_found": False,
                                     "products": [{"name": "Guidewire", "vendor": "Guidewire Inc", "claims": []},
                                                  {"name": "Snapsheet", "vendor": "", "claims": []}]},
            "SEG-APP-3-standalone": {"segment_id": "SEG-APP-3-standalone", "application_id": "APP-3",
                                     "framing": "standalone", "no_viable_alternative_found": True, "products": []},
        },
        "phase2_agenda": [
            {"application_id": "APP-2", "field_label": "Annual license cost", "category": "Withheld-Confidential"},
            {"application_id": "APP-3", "field_label": "Business criticality", "category": "Deferred-until-award"},
        ],
        "ingestion_collisions": [],
        "branch_failures": [],
        "gate_decisions": {"gate_1_rubric_signoff": {"decision": "approved", "item_count": 0}},
    }
    base.update(overrides)
    return base


# --- structure & ordering -------------------------------------------------


def test_applications_are_ordered_by_tim_e_score_desc_with_unscored_last():
    report = report_service.build_report(_state())
    assert [a["application_id"] for a in report["applications"]] == ["APP-1", "APP-2", "APP-3"]


def test_every_top_level_section_is_present():
    report = report_service.build_report(_state())
    assert set(report) >= {"run_id", "data_sensitivity", "portfolio_summary", "applications", "run_integrity"}
    assert report["run_id"] == "run-1"


# --- per-application assembly -------------------------------------------


def test_redundancy_counterpart_is_resolved_from_whichever_side_the_app_is_on():
    report = report_service.build_report(_state())
    app1 = next(a for a in report["applications"] if a["application_id"] == "APP-1")
    app2 = next(a for a in report["applications"] if a["application_id"] == "APP-2")
    assert app1["redundancy"][0]["counterpart"] == "APP-2"
    assert app2["redundancy"][0]["counterpart"] == "APP-1"
    assert app1["redundancy"][0]["mandatory_review"] is True  # from the recommendation half


def test_market_alternatives_are_aggregated_and_deduped_across_a_tiered_apps_segments():
    report = report_service.build_report(_state())
    app1 = next(a for a in report["applications"] if a["application_id"] == "APP-1")
    market = app1["market_alternatives"]
    assert [p["name"] for p in market["products"]] == ["Guidewire", "Snapsheet"]  # Guidewire not doubled
    assert market["product_count"] == 2
    assert market["no_viable_alternative_found"] is False


def test_a_no_viable_alternative_segment_is_reported_as_such():
    report = report_service.build_report(_state())
    app3 = next(a for a in report["applications"] if a["application_id"] == "APP-3")
    assert app3["market_alternatives"]["no_viable_alternative_found"] is True
    assert app3["market_alternatives"]["product_count"] == 0


def test_cost_outlier_and_its_explainability_are_carried_through():
    report = report_service.build_report(_state())
    app1 = next(a for a in report["applications"] if a["application_id"] == "APP-1")
    assert app1["cost_outlier"]["direction"] == "high"
    assert app1["cost_outlier"]["explainability"]["needs_review"] is True


def test_flags_surface_fallback_pending_reviews_and_withheld_fields():
    report = report_service.build_report(_state())
    by_id = {a["application_id"]: a for a in report["applications"]}

    assert any("structured bullets" in f for f in by_id["APP-2"]["flags"])
    assert any("gate 5" in f for f in by_id["APP-2"]["flags"])
    assert any("gate 3" in f for f in by_id["APP-1"]["flags"])
    assert any("gate 4" in f for f in by_id["APP-1"]["flags"])
    assert any("Phase 2" in f and "Annual license cost" in f for f in by_id["APP-2"]["flags"])


def test_an_application_with_no_narrative_is_flagged():
    report = report_service.build_report(_state())
    app3 = next(a for a in report["applications"] if a["application_id"] == "APP-3")
    assert app3["narrative"] is None
    assert any("No narrative" in f for f in app3["flags"])


# --- portfolio summary -------------------------------------------------


def test_portfolio_summary_counts_decisions_typologies_and_flags():
    summary = report_service.build_report(_state())["portfolio_summary"]
    assert summary["application_count"] == 3
    assert summary["time_decisions"] == {"Migrate": 1, "Eliminate": 1, "Insufficient Data": 1}
    assert summary["cots_replace_candidates"] == 1
    assert summary["cots_replace_threshold"] == gp.COTS_REPLACE_THRESHOLD
    assert summary["redundancy_typologies"] == {"Scale-Tiered Overlap": 1}
    assert summary["cost_outliers_flagged"] == 1
    assert summary["market_segments_researched"] == 3
    assert summary["no_viable_alternative_segments"] == 1
    assert summary["phase2_discovery_items"] == 2


# --- run integrity & empty run --------------------------------------


def test_run_integrity_passes_through_collisions_failures_and_gate_decisions():
    state = _state(
        ingestion_collisions=[{"application_id": "APP-9", "occurrences": 2}],
        branch_failures=[{"branch_kind": "market", "subject_id": "SEG-X", "error": "RuntimeError: boom"}],
    )
    integrity = report_service.build_report(state)["run_integrity"]
    assert integrity["ingestion_collisions"] == [{"application_id": "APP-9", "occurrences": 2}]
    assert integrity["branch_failures"][0]["subject_id"] == "SEG-X"
    assert "gate_1_rubric_signoff" in integrity["gate_decisions"]


def test_an_empty_run_still_returns_the_full_shape():
    report = report_service.build_report({"run_id": "run-empty"})
    assert report["applications"] == []
    assert report["portfolio_summary"]["application_count"] == 0
    assert report["portfolio_summary"]["time_decisions"] == {}
    assert report["run_integrity"] == {"ingestion_collisions": [], "branch_failures": [], "gate_decisions": {}}
