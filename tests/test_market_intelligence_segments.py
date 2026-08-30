"""Segment construction -- CLAUDE.md sections 8, 9. Purely deterministic."""

from __future__ import annotations

from app.market_intelligence import segments as seg


def _app(application_id, l3="Supplier Onboarding", **overrides):
    base = {
        "application_id": application_id,
        "business_capability_l1": "Supply Chain",
        "business_capability_l2": "Source to Contract",
        "business_capability_l3": l3,
    }
    base.update(overrides)
    return base


def _profile(fte_count=None, cost_per_fte=None):
    return {"scale_usage": {"fte_count": fte_count}, "cost": {"cost_per_fte": cost_per_fte}}


def _verdict(typology, a, b, cluster_id="CL-1"):
    return {"typology": typology, "cluster_id": cluster_id, "application_id_a": a, "application_id_b": b}


# --- True Duplicate: exactly one segment, for the retained (heavier) app --


def test_true_duplicate_produces_one_segment_for_the_heavier_app():
    verdicts = [_verdict("True Duplicate", "A", "B")]
    applications = [_app("A"), _app("B")]
    profiles = {"A": _profile(fte_count=20), "B": _profile(fte_count=5)}

    result = seg.build_segments(verdicts, applications, profiles)

    assert len(result) == 1
    assert result[0].application_id == "A"
    assert result[0].framing == seg.STANDALONE


def test_true_duplicate_never_produces_a_segment_for_the_retired_app():
    verdicts = [_verdict("True Duplicate", "A", "B")]
    applications = [_app("A"), _app("B")]
    profiles = {"A": _profile(fte_count=20), "B": _profile(fte_count=5)}

    result = seg.build_segments(verdicts, applications, profiles)

    assert all(s.application_id != "B" for s in result)


def test_true_duplicate_falls_back_to_cost_per_fte_when_fte_counts_tie():
    verdicts = [_verdict("True Duplicate", "A", "B")]
    applications = [_app("A"), _app("B")]
    profiles = {"A": _profile(fte_count=10, cost_per_fte=5_000), "B": _profile(fte_count=10, cost_per_fte=50_000)}

    result = seg.build_segments(verdicts, applications, profiles)

    assert result[0].application_id == "B"  # higher cost-per-fte wins the tie-break


# --- Scale-Tiered Overlap: two tier-framed segments --------------------------


def test_scale_tiered_overlap_produces_two_differently_framed_segments():
    verdicts = [_verdict("Scale-Tiered Overlap", "A", "B")]
    applications = [_app("A"), _app("B")]
    profiles = {"A": _profile(fte_count=50), "B": _profile(fte_count=3)}

    result = seg.build_segments(verdicts, applications, profiles)

    by_id = {s.application_id: s for s in result}
    assert len(result) == 2
    assert by_id["A"].framing == seg.TIER_ENTERPRISE
    assert by_id["B"].framing == seg.TIER_LIGHT
    assert by_id["A"].seed_query != by_id["B"].seed_query  # differently framed, per CLAUDE.md section 8


# --- Partial/Component Overlap: two segments, neutral framing --------------


def test_partial_overlap_produces_a_segment_for_each_application():
    verdicts = [_verdict("Partial/Component Overlap", "A", "B")]
    applications = [_app("A"), _app("B")]
    result = seg.build_segments(verdicts, applications, {})

    by_id = {s.application_id: s for s in result}
    assert len(result) == 2
    assert by_id["A"].framing == seg.PARTIAL_OVERLAP
    assert by_id["B"].framing == seg.PARTIAL_OVERLAP


# --- Distinct: individually, one segment per app -----------------------------


def test_distinct_produces_a_standalone_segment_for_each_application():
    verdicts = [_verdict("Distinct", "A", "B")]
    applications = [_app("A"), _app("B")]
    result = seg.build_segments(verdicts, applications, {})

    assert len(result) == 2
    assert all(s.framing == seg.STANDALONE for s in result)


# --- Indeterminate / Adjudication Failed: deferred, no segment ---------------


def test_indeterminate_produces_no_segment():
    verdicts = [_verdict("Indeterminate — Withheld Data", "A", "B")]
    applications = [_app("A"), _app("B")]
    assert seg.build_segments(verdicts, applications, {}) == []


def test_adjudication_failed_produces_no_segment():
    verdicts = [_verdict("Adjudication Failed", "A", "B")]
    applications = [_app("A"), _app("B")]
    assert seg.build_segments(verdicts, applications, {}) == []


# --- unclustered singleton applications: standalone, like Distinct ----------


def test_an_application_with_no_verdict_at_all_still_gets_a_standalone_segment():
    """Blocking drops singleton clusters -- an app with no capability
    peers never reaches the adjudicator, but section 8's own purpose
    (COTS discovery across the portfolio) does not stop at the redundant
    subset."""
    applications = [_app("A")]
    result = seg.build_segments([], applications, {})

    assert len(result) == 1
    assert result[0].application_id == "A"
    assert result[0].framing == seg.STANDALONE
    assert result[0].cluster_id is None
    assert result[0].typology is None


# --- priority and dedup across a multi-member cluster -----------------------


def test_a_real_verdict_always_claims_an_application_ahead_of_a_deferred_one():
    """3-member cluster: A-B is Distinct, A-C is Indeterminate. A must
    end up covered (not deferred), regardless of dict/list iteration
    order."""
    verdicts = [
        _verdict("Indeterminate — Withheld Data", "A", "C"),
        _verdict("Distinct", "A", "B"),
    ]
    applications = [_app("A"), _app("B"), _app("C")]
    result = seg.build_segments(verdicts, applications, {})

    covered_ids = {s.application_id for s in result}
    assert "A" in covered_ids
    assert "C" not in covered_ids  # C has no other verdict claiming it


def test_true_duplicate_retirement_takes_priority_over_a_distinct_verdict_for_the_same_app():
    """3-member cluster: B is retired via True Duplicate with C, and B
    also appears in a Distinct verdict with A -- listed *before* the
    True Duplicate verdict, so this only passes if processing order is
    priority-based (True Duplicate first), not list order."""
    verdicts = [
        _verdict("Distinct", "A", "B"),
        _verdict("True Duplicate", "B", "C"),
    ]
    applications = [_app("A"), _app("B"), _app("C")]
    profiles = {"B": _profile(fte_count=5), "C": _profile(fte_count=50)}  # C retained, B retired

    result = seg.build_segments(verdicts, applications, profiles)

    covered_ids = {s.application_id for s in result}
    assert "B" not in covered_ids  # retired -- must not pick up a segment via the Distinct pairing
    assert "A" in covered_ids
    assert "C" in covered_ids


def test_an_application_is_never_given_two_segments_across_multiple_pair_verdicts():
    """3-member cluster, A appears in two Distinct pairs (A-B, A-C) --
    A must get exactly one segment, not two."""
    verdicts = [_verdict("Distinct", "A", "B"), _verdict("Distinct", "A", "C")]
    applications = [_app("A"), _app("B"), _app("C")]
    result = seg.build_segments(verdicts, applications, {})

    assert sum(1 for s in result if s.application_id == "A") == 1
    assert len(result) == 3  # A, B, C each exactly once


# --- capability label and seed query -----------------------------------------


def test_capability_label_prefers_l3_falling_back_to_l2_then_l1():
    assert seg._capability_label(_app("A", l3="Supplier Onboarding")) == "Supplier Onboarding"
    assert seg._capability_label(_app("A", l3=None)) == "Source to Contract"
    assert seg._capability_label(_app("A", l3=None, business_capability_l2=None)) == "Supply Chain"


def test_a_verdict_referencing_an_application_id_not_in_the_portfolio_is_skipped():
    verdicts = [_verdict("Distinct", "A", "GHOST")]
    applications = [_app("A")]
    result = seg.build_segments(verdicts, applications, {})
    assert all(s.application_id != "GHOST" for s in result)


def test_segment_ids_are_deterministic_and_distinguish_framing():
    verdicts = [_verdict("Scale-Tiered Overlap", "A", "B")]
    applications = [_app("A"), _app("B")]
    profiles = {"A": _profile(fte_count=50), "B": _profile(fte_count=3)}
    result = seg.build_segments(verdicts, applications, profiles)
    segment_ids = {s.segment_id for s in result}
    assert len(segment_ids) == 2  # unique per (application, framing)


# --- self_match_terms: for the agent's self-match filter --------------------


def test_self_match_terms_include_application_name_and_tech_stack():
    applications = [_app("A", application_name="Internal Onboarding Tool")]
    profiles = {"A": {"technical": {"technology_stack_tokens": ["coupa", "sap ariba"]}}}
    result = seg.build_segments([], applications, profiles)
    terms = set(result[0].self_match_terms)
    assert "Internal Onboarding Tool" in terms
    assert "coupa" in terms
    assert "sap ariba" in terms


def test_self_match_terms_never_appear_in_the_seed_query():
    """The seed query must never leak the client's own product/vendor
    names into an external search."""
    applications = [_app("A", application_name="Internal Onboarding Tool")]
    profiles = {"A": {"technical": {"technology_stack_tokens": ["coupa"]}}}
    result = seg.build_segments([], applications, profiles)
    assert "Internal Onboarding Tool" not in result[0].seed_query
    assert "coupa" not in result[0].seed_query.lower()


def test_self_match_terms_are_deduped_and_blanks_dropped():
    applications = [_app("A", application_name="Tool")]
    profiles = {"A": {"technical": {"technology_stack_tokens": ["coupa", "coupa", "", "SAP"]}}}
    result = seg.build_segments([], applications, profiles)
    assert result[0].self_match_terms.count("coupa") == 1
    assert "" not in result[0].self_match_terms


def test_missing_profile_yields_empty_self_match_terms_from_the_stack():
    applications = [_app("A", application_name="Tool")]
    result = seg.build_segments([], applications, {})
    assert result[0].self_match_terms == ("Tool",)


def test_as_dict_is_json_serializable():
    import json

    result = seg.build_segments([], [_app("A")], {})
    json.dumps(result[0].as_dict())
