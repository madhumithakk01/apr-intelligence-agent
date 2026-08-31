"""Redundancy typology regression -- the SPEC.md section 9 table, end to end.

test_redundancy_adjudicator.py covers the ensemble mechanics and
test_recommendation_policy.py covers the four non-compensatory gates in
isolation. This file runs one realistic two-application scenario per
typology through the whole deterministic redundancy chain --

    build_profile
      -> adjudicator.adjudicate_pair
      -> recommendation_policy.evaluate
      -> market_intelligence.segments.build_segments

-- and asserts the section 9 table row for row: trigger -> typology ->
recommendation -> market-research cardinality -> gate-3 routing ->
Phase 2 flag.

Deterministic: the ensemble vote is stubbed, so no LLM call is made
(and the Indeterminate case asserts the vote is never even reached).
"""

from __future__ import annotations

import pytest

from app.llm.providers import DataSensitivity
from app.market_intelligence import segments as seg
from app.redundancy import adjudicator as adj
from app.redundancy import recommendation_policy as rp
from app.redundancy.profile_builder import build_profile

_CLUSTER = "CL-TEST"


# --- scenario builders ------------------------------------------------


def _app(app_id, *, l1="Operations", l2="Claims", l3="Claims Intake", fte=50,
         criticality="medium", security="Confidential", stability="high",
         stack="Java, Oracle", fte_cost=200_000, description="claims intake handling",
         **overrides):
    row = {
        "application_id": app_id,
        "application_name": app_id,
        "business_capability_l1": l1,
        "business_capability_l2": l2,
        "business_capability_l3": l3,
        "application_description": description,
        "fte_count": fte,
        "business_criticality": criticality,
        "application_security_level": security,
        "application_stability": stability,
        "technology_stack": stack,
        "maintainability": "medium",
        "functional_redundancy": "cannot say",
        "annual_fte_cost": fte_cost,
    }
    row.update(overrides)
    return row


def _stub_unanimous(monkeypatch, typology):
    monkeypatch.setattr(
        adj, "_call_llm_typology", lambda *a, **k: adj.EnsembleVote(typology=typology, rationale="stub")
    )


def _forbid_llm(monkeypatch):
    monkeypatch.setattr(adj, "_call_llm_typology", lambda *a, **k: pytest.fail("no ensemble call expected"))


def _chain(app_a, app_b):
    profile_a, profile_b = build_profile(app_a), build_profile(app_b)
    verdict = adj.adjudicate_pair(
        profile_a, profile_b, cluster_id=_CLUSTER, data_sensitivity=DataSensitivity.SYNTHETIC
    )
    recommendation = rp.evaluate(verdict, profile_a, profile_b)
    return profile_a, profile_b, verdict, recommendation


def _segments(verdict, apps, profiles):
    return seg.build_segments(
        [verdict.as_dict()],
        apps,
        {p.application_id: p.as_dict() for p in profiles},
    )


# --- True Duplicate ------------------------------------------------
# Same capability, comparable scale/cost-per-unit -> consolidate, retire
# one; market research once, for the retained application.


def test_true_duplicate_consolidates_and_researches_the_retained_app_once(monkeypatch):
    _stub_unanimous(monkeypatch, adj.TRUE_DUPLICATE)
    a = _app("TD-A", fte=50, fte_cost=250_000)
    b = _app("TD-B", fte=48, fte_cost=250_000)

    profile_a, profile_b, verdict, rec = _chain(a, b)

    assert verdict.typology == adj.TRUE_DUPLICATE
    assert verdict.resolution == "unanimous"
    assert verdict.mandatory_review is True  # any True-Duplicate majority routes to gate 3

    assert rec.recommendation.startswith("Consolidate; retire one application.")
    assert rec.consolidation_blocked_by is None
    assert rec.mandatory_review is True
    assert rec.phase2_discovery is False

    segments = _segments(verdict, [a, b], [profile_a, profile_b])
    assert len(segments) == 1
    assert segments[0].framing == seg.STANDALONE
    assert segments[0].application_id == "TD-A"  # the heavier, retained side


def test_true_duplicate_with_a_classification_mismatch_is_blocked_from_consolidating(monkeypatch):
    _stub_unanimous(monkeypatch, adj.TRUE_DUPLICATE)
    a = _app("TD-A", security="Confidential")
    b = _app("TD-B", fte=48, security="Internal use only")

    _, _, _verdict, rec = _chain(a, b)

    assert rec.consolidation_blocked_by == "classification_mismatch"
    assert rec.recommendation.startswith("Retain both")
    assert rec.phase2_discovery is True


# --- Scale-Tiered Overlap ---------------------------------------
# Same capability, materially different scale -> retain both as tiers, or
# migrate the light tier if normalized cost + feasibility support it;
# market research separately per tier.


def test_scale_tiered_overlap_migration_is_supported_and_researched_per_tier(monkeypatch):
    _stub_unanimous(monkeypatch, adj.SCALE_TIERED_OVERLAP)
    heavy = _app("STO-H", fte=100, fte_cost=300_000, criticality="medium", stability="very high")
    light = _app("STO-L", fte=3, fte_cost=90_000, criticality="low", stability="medium")

    profile_h, profile_l, verdict, rec = _chain(heavy, light)

    assert verdict.typology == adj.SCALE_TIERED_OVERLAP
    assert rec.recommendation.startswith("Migrate the lighter tier onto the heavier platform.")
    assert rec.consolidation_blocked_by is None
    assert rec.mandatory_review is True  # STO recommending consolidation -> gate 3, regardless of confidence
    assert rec.phase2_discovery is False

    segments = _segments(verdict, [heavy, light], [profile_h, profile_l])
    assert {s.framing for s in segments} == {seg.TIER_ENTERPRISE, seg.TIER_LIGHT}
    assert {s.application_id: s.framing for s in segments} == {
        "STO-H": seg.TIER_ENTERPRISE,
        "STO-L": seg.TIER_LIGHT,
    }


def test_scale_tiered_overlap_high_criticality_light_tier_is_kept_as_a_tier(monkeypatch):
    _stub_unanimous(monkeypatch, adj.SCALE_TIERED_OVERLAP)
    heavy = _app("STO-H", fte=100, fte_cost=300_000, stability="low")
    light = _app("STO-L", fte=3, fte_cost=400_000, criticality="very high")

    profile_h, profile_l, verdict, rec = _chain(heavy, light)

    assert rec.consolidation_blocked_by == "criticality_ceiling"
    assert rec.recommendation.startswith("Retain both as differentiated tiers")
    # still two research targets, one per tier
    assert len(_segments(verdict, [heavy, light], [profile_h, profile_l])) == 2


# --- Partial / Component Overlap -----------------------------
# L1/L2 match, L3 diverges -> retain both, log for Phase 2; research
# separately, one segment per application.


def test_partial_component_overlap_retains_both_and_logs_phase2(monkeypatch):
    _stub_unanimous(monkeypatch, adj.PARTIAL_COMPONENT_OVERLAP)
    a = _app("PC-A", l3="Claims Intake")
    b = _app("PC-B", fte=40, l3="Claims Adjudication")

    profile_a, profile_b, verdict, rec = _chain(a, b)

    assert rec.recommendation == "Retain both; log for Phase 2 scoping."
    assert rec.consolidation_blocked_by is None
    assert rec.phase2_discovery is True
    assert rec.mandatory_review is False  # a Partial-Overlap majority is not auto-reviewed

    segments = _segments(verdict, [a, b], [profile_a, profile_b])
    assert [s.framing for s in segments] == [seg.PARTIAL_OVERLAP, seg.PARTIAL_OVERLAP]
    assert {s.application_id for s in segments} == {"PC-A", "PC-B"}


# --- Distinct -------------------------------------------------
# Superficial match only -> no action; research individually.


def test_distinct_is_no_action_and_researched_individually(monkeypatch):
    _stub_unanimous(monkeypatch, adj.DISTINCT)
    a = _app("DI-A", l2="Claims", l3="Claims Intake")
    b = _app("DI-B", fte=30, l2="Billing", l3="Invoice Runs")

    profile_a, profile_b, verdict, rec = _chain(a, b)

    assert rec.recommendation == "No action."
    assert rec.phase2_discovery is False
    assert rec.mandatory_review is False

    segments = _segments(verdict, [a, b], [profile_a, profile_b])
    assert [s.framing for s in segments] == [seg.STANDALONE, seg.STANDALONE]
    assert {s.application_id for s in segments} == {"DI-A", "DI-B"}


# --- Indeterminate -- Withheld Data --------------------------
# A verdict-critical field is withheld -> no verdict, mandatory review,
# Phase 2 discovery item; market research deferred (no segment).


def test_withheld_cost_is_indeterminate_with_no_ensemble_call(monkeypatch):
    _forbid_llm(monkeypatch)
    a = _app("IN-A")
    b = _app("IN-B", fte=48, fte_cost=None)  # no cost fields -> cost_per_fte is None

    profile_a, profile_b, verdict, rec = _chain(a, b)

    assert verdict.typology == adj.INDETERMINATE_WITHHELD_DATA
    assert verdict.resolution == "deterministic_withheld"
    assert verdict.votes == []
    assert verdict.mandatory_review is True

    assert rec.recommendation.startswith("No verdict")
    assert rec.mandatory_review is True
    assert rec.phase2_discovery is True

    assert _segments(verdict, [a, b], [profile_a, profile_b]) == []


@pytest.mark.parametrize(
    "field_overrides",
    [
        {"application_security_level": None},
        {"business_criticality": None},
    ],
)
def test_other_withheld_verdict_critical_fields_are_also_indeterminate(monkeypatch, field_overrides):
    _forbid_llm(monkeypatch)
    _, _, verdict, _rec = _chain(_app("IN-A"), _app("IN-B", fte=48, **field_overrides))
    assert verdict.typology == adj.INDETERMINATE_WITHHELD_DATA


# --- the section 9 "Market research" column, all five rows -------


def test_market_research_cardinality_matches_the_section_9_table(monkeypatch):
    expected = {
        adj.TRUE_DUPLICATE: 1,
        adj.SCALE_TIERED_OVERLAP: 2,
        adj.PARTIAL_COMPONENT_OVERLAP: 2,
        adj.DISTINCT: 2,
    }
    for typology, count in expected.items():
        _stub_unanimous(monkeypatch, typology)
        a = _app("X-A", fte=100, fte_cost=300_000)
        b = _app("X-B", fte=3, fte_cost=400_000)
        profile_a, profile_b, verdict, _rec = _chain(a, b)
        assert len(_segments(verdict, [a, b], [profile_a, profile_b])) == count, typology

    _forbid_llm(monkeypatch)
    a = _app("X-A")
    b = _app("X-B", fte=3, fte_cost=None)
    profile_a, profile_b, verdict, _rec = _chain(a, b)
    assert _segments(verdict, [a, b], [profile_a, profile_b]) == []  # Indeterminate -> deferred
