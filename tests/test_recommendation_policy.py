"""Non-compensatory recommendation policy -- CLAUDE.md sections 9, 10."""

from __future__ import annotations

from app.redundancy import adjudicator as adj
from app.redundancy import recommendation_policy as rp
from app.redundancy.profile_builder import build_profile


def _profile(application_id="A", **overrides):
    base = {
        "application_id": application_id,
        "business_capability_l1": "Finance",
        "business_capability_l2": "Record to Report",
        "fte_count": 10,
        "business_criticality": "high",  # canonical label -- kernel.score_qualitative_label recognizes it
        "annual_fte_cost": 100_000,
        "annual_license_cost": 50_000,
        "annual_infrastructure_cost": 20_000,
        "other_costs": 10_000,
        "application_security_level": "Confidential",
        "application_stability": "very high",  # canonical label
        "technology_stack": "SAP, Oracle DB",
        "maintainability": "Simple",
    }
    base.update(overrides)
    return build_profile(base)


def _verdict(typology, *, mandatory_review=False, resolution="unanimous"):
    return adj.AdjudicationVerdict(
        cluster_id="CL-1", application_id_a="A", application_id_b="B",
        typology=typology, resolution=resolution, votes=[], mandatory_review=mandatory_review,
        rationale="r",
    )


# --- fixed recommendations for non-consolidation-candidate typologies ------


def test_partial_component_overlap_is_retain_both_and_logged_for_phase2():
    result = rp.evaluate(_verdict(adj.PARTIAL_COMPONENT_OVERLAP), _profile("A"), _profile("B"))
    assert "Retain both" in result.recommendation
    assert result.phase2_discovery is True
    assert result.consolidation_blocked_by is None


def test_distinct_is_no_action_and_not_logged():
    result = rp.evaluate(_verdict(adj.DISTINCT), _profile("A"), _profile("B"))
    assert result.recommendation == "No action."
    assert result.phase2_discovery is False
    assert result.mandatory_review is False


def test_indeterminate_is_no_verdict_mandatory_review_and_phase2():
    verdict = _verdict(adj.INDETERMINATE_WITHHELD_DATA, mandatory_review=True, resolution="deterministic_withheld")
    result = rp.evaluate(verdict, _profile("A"), _profile("B"))
    assert "No verdict" in result.recommendation
    assert result.mandatory_review is True
    assert result.phase2_discovery is True


def test_adjudication_failed_is_no_verdict_mandatory_review_and_phase2():
    verdict = _verdict(adj.ADJUDICATION_FAILED, mandatory_review=True, resolution="failed")
    result = rp.evaluate(verdict, _profile("A"), _profile("B"))
    assert "did not complete" in result.recommendation
    assert result.mandatory_review is True
    assert result.phase2_discovery is True


# --- gate 1: data classification -----------------------------------------


def test_classification_mismatch_blocks_true_duplicate_consolidation():
    a = _profile("A", application_security_level="Confidential")
    b = _profile("B", application_security_level="Public")
    result = rp.evaluate(_verdict(adj.TRUE_DUPLICATE), a, b)
    assert result.consolidation_blocked_by == "classification_mismatch"
    assert "Retain both" in result.recommendation
    assert "Consolidate" not in result.recommendation


def test_classification_mismatch_blocks_scale_tiered_overlap_migration():
    a = _profile("A", application_security_level="Confidential", fte_count=20)
    b = _profile("B", application_security_level="Public", fte_count=5)
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), a, b)
    assert result.consolidation_blocked_by == "classification_mismatch"
    assert "differentiated tiers" in result.recommendation


def test_matching_classification_does_not_trigger_gate_1():
    a = _profile("A", application_security_level="Confidential")
    b = _profile("B", application_security_level="confidential")  # case-insensitive match
    result = rp.evaluate(_verdict(adj.TRUE_DUPLICATE), a, b)
    assert result.consolidation_blocked_by != "classification_mismatch"


# --- gate 2: criticality ceiling --------------------------------------------


def test_high_criticality_light_tier_onto_unproven_heavy_platform_is_blocked():
    heavy = _profile("Heavy", fte_count=50, business_criticality="Low", application_stability="low")
    light = _profile("Light", fte_count=5, business_criticality="very high")
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), heavy, light)
    assert result.consolidation_blocked_by == "criticality_ceiling"


def test_high_criticality_light_tier_onto_proven_heavy_platform_is_not_blocked_by_gate_2():
    heavy = _profile("Heavy", fte_count=50, business_criticality="Low", application_stability="very high")
    light = _profile("Light", fte_count=5, business_criticality="very high")
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), heavy, light)
    assert result.consolidation_blocked_by != "criticality_ceiling"


def test_unreadable_criticality_label_blocks_rather_than_guesses():
    """This branch has no calibrated-rubric access (feat/qualitative-
    scoring is not an ancestor) -- free text that isn't one of the five
    fixed canonical labels must never be silently treated as low."""
    heavy = _profile("Heavy", fte_count=50, business_criticality="Low", application_stability="Stable")
    light = _profile("Light", fte_count=5, business_criticality="Somewhat important")  # not a canonical label
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), heavy, light)
    assert result.consolidation_blocked_by == "criticality_or_stability_unreadable"


# --- gate 3: normalized cost-vs-scale ---------------------------------------


def test_migration_blocked_when_light_tier_cost_per_fte_is_not_higher():
    heavy = _profile(
        "Heavy", fte_count=50, business_criticality="Low", application_stability="very high",
        annual_fte_cost=500_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0,
    )
    light = _profile(
        "Light", fte_count=5, business_criticality="Low",
        annual_fte_cost=5_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0,
    )
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), heavy, light)
    assert result.consolidation_blocked_by == "cost_does_not_justify"


def test_migration_supported_when_light_tier_cost_per_fte_is_higher():
    heavy = _profile(
        "Heavy", fte_count=50, business_criticality="Low", application_stability="very high",
        annual_fte_cost=500_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0,
    )
    light = _profile(
        "Light", fte_count=2, business_criticality="Low",
        annual_fte_cost=100_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0,
    )
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), heavy, light)
    assert result.consolidation_blocked_by is None
    assert "Migrate" in result.recommendation


# --- gate 4: technical feasibility never blocks -----------------------------


def test_technical_feasibility_never_blocks_only_annotates():
    # B has a smaller fte_count than A -- higher cost-per-FTE -- so it
    # clears the cost gate as the migration candidate, leaving only the
    # feasibility gate to reach.
    a = _profile("A", technology_stack="SAP")
    b = _profile("B", fte_count=2, technology_stack="Workday")  # no shared stack
    result = rp.evaluate(_verdict(adj.TRUE_DUPLICATE), a, b)
    assert result.consolidation_blocked_by is None
    assert "shared technology stack" in result.recommendation.lower() or "no shared" in result.recommendation.lower()


def test_shared_stack_produces_a_different_note_than_no_shared_stack():
    a = _profile("A", technology_stack="SAP")
    shared = rp.evaluate(_verdict(adj.TRUE_DUPLICATE), a, _profile("B", fte_count=2, technology_stack="SAP"))
    unshared = rp.evaluate(_verdict(adj.TRUE_DUPLICATE), a, _profile("B", fte_count=2, technology_stack="Workday"))
    assert shared.recommendation != unshared.recommendation


# --- gate ordering: classification blocks even when criticality would too --


def test_gates_apply_in_order_classification_wins_first():
    a = _profile("A", application_security_level="Confidential", fte_count=50, business_criticality="Low", application_stability="Unstable")
    b = _profile("B", application_security_level="Public", fte_count=5, business_criticality="very high")
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP), a, b)
    assert result.consolidation_blocked_by == "classification_mismatch"  # not criticality_ceiling


# --- gate 3 (report gate, section 10) finalization --------------------------


def test_true_duplicate_recommending_consolidation_is_always_reviewed():
    result = rp.evaluate(
        _verdict(adj.TRUE_DUPLICATE, mandatory_review=True), _profile("A"), _profile("B", fte_count=2)
    )
    assert result.consolidation_blocked_by is None
    assert result.mandatory_review is True


def test_scale_tiered_overlap_recommending_migration_is_always_reviewed():
    heavy = _profile("Heavy", fte_count=50, business_criticality="Low", application_stability="very high",
                      annual_fte_cost=500_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0)
    light = _profile("Light", fte_count=2, business_criticality="Low",
                      annual_fte_cost=100_000, annual_license_cost=0, annual_infrastructure_cost=0, other_costs=0)
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP, mandatory_review=False), heavy, light)
    assert result.mandatory_review is True  # gate-3 rule applies even though the ensemble itself didn't flag it


def test_scale_tiered_overlap_blocked_from_consolidating_is_not_forced_into_review_by_this_rule():
    """The Scale-Tiered-Overlap-always-reviewed rule is specifically for
    the consolidation-recommending outcome -- a blocked one inherits
    only the verdict's own mandatory_review, not an automatic True."""
    a = _profile("A", application_security_level="Confidential")
    b = _profile("B", application_security_level="Public")
    result = rp.evaluate(_verdict(adj.SCALE_TIERED_OVERLAP, mandatory_review=False), a, b)
    assert result.mandatory_review is False


# --- serialization ------------------------------------------------------------


def test_as_dict_is_json_serializable():
    import json

    result = rp.evaluate(_verdict(adj.DISTINCT), _profile("A"), _profile("B"))
    json.dumps(result.as_dict())
