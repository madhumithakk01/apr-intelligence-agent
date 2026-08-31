"""Multi-axis profile building -- SPEC.md sections 5 and 9."""

from __future__ import annotations

import json

from app.redundancy import profile_builder as pb


def _app(application_id="A", **overrides):
    base = {"application_id": application_id}
    base.update(overrides)
    return base


# --- functional axis -----------------------------------------------------


def test_functional_axis_carries_capability_tags_verbatim():
    profile = pb.build_profile(
        _app(business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL")
    )
    assert profile.functional.capability_l1 == "Finance"
    assert profile.functional.capability_l2 == "R2R"
    assert profile.functional.capability_l3 == "GL"


def test_missing_capability_tags_are_none_not_empty_string():
    profile = pb.build_profile(_app())
    assert profile.functional.capability_l1 is None


def test_description_is_tokenized_lowercase_deduped_and_stopwords_dropped():
    profile = pb.build_profile(_app(application_description="The Customer Relationship Management and the CRM system"))
    tokens = profile.functional.description_tokens
    assert "the" not in tokens
    assert "and" not in tokens
    assert "customer" in tokens
    assert "crm" in tokens
    assert tokens == tuple(sorted(set(tokens)))  # deduped and sorted


def test_missing_description_yields_empty_tokens():
    profile = pb.build_profile(_app())
    assert profile.functional.description_tokens == ()


# --- scale/usage axis ------------------------------------------------------


def test_scale_usage_axis_fields():
    profile = pb.build_profile(_app(fte_count=12, usage_adoption="High", business_criticality="Strategic"))
    assert profile.scale_usage.fte_count == 12
    assert profile.scale_usage.usage_adoption == "High"
    assert profile.scale_usage.business_criticality == "Strategic"


def test_missing_scale_usage_fields_are_none():
    profile = pb.build_profile(_app())
    assert profile.scale_usage.fte_count is None
    assert profile.scale_usage.usage_adoption is None
    assert profile.scale_usage.business_criticality is None


# --- cost axis: normalized, never a raw total -------------------------------


def test_cost_per_fte_is_normalized_not_a_raw_total():
    profile = pb.build_profile(
        _app(fte_count=10, annual_fte_cost=100_000, annual_license_cost=50_000,
             annual_infrastructure_cost=20_000, other_costs=10_000)
    )
    assert profile.cost.cost_per_fte == 18_000.0  # 180,000 / 10
    assert profile.cost.is_complete is True


def test_cost_is_none_when_fte_count_missing():
    profile = pb.build_profile(_app(annual_fte_cost=100_000))
    assert profile.cost.cost_per_fte is None
    assert profile.cost.is_complete is False


def test_cost_is_none_when_fte_count_is_zero():
    profile = pb.build_profile(_app(fte_count=0, annual_fte_cost=100_000))
    assert profile.cost.cost_per_fte is None


def test_cost_is_none_when_every_cost_component_missing():
    profile = pb.build_profile(_app(fte_count=10))
    assert profile.cost.cost_per_fte is None
    assert profile.cost.is_complete is False


def test_partial_cost_is_computed_but_flagged_incomplete():
    """A partial sum is still informative but must never be mistaken for
    a complete one."""
    profile = pb.build_profile(_app(fte_count=10, annual_fte_cost=100_000))
    assert profile.cost.cost_per_fte == 10_000.0  # 100,000 / 10, partial
    assert profile.cost.is_complete is False


def test_a_withheld_cost_cell_represented_as_none_is_never_treated_as_zero():
    """Mirrors SPEC.md section 2/4 bug 6: a withheld cost must never
    silently coerce into 0 and quietly deflate cost_per_fte."""
    all_four = dict(fte_count=10, annual_fte_cost=100_000, annual_license_cost=50_000, annual_infrastructure_cost=20_000)
    withheld = pb.build_profile(_app(**all_four, other_costs=None))
    known_zero = pb.build_profile(_app(**all_four, other_costs=0))
    assert withheld.cost.cost_per_fte == known_zero.cost.cost_per_fte  # None omitted == 0 included, here
    assert withheld.cost.is_complete is False  # one component genuinely unknown
    assert known_zero.cost.is_complete is True  # 0 is a known value, None is not


# --- risk/classification axis -----------------------------------------------


def test_risk_classification_axis_fields():
    profile = pb.build_profile(
        _app(application_security_level="Confidential", application_stability="Stable", availability="Always available")
    )
    assert profile.risk_classification.application_security_level == "Confidential"
    assert profile.risk_classification.application_stability == "Stable"
    assert profile.risk_classification.availability == "Always available"


# --- technical axis ----------------------------------------------------------


def test_technology_stack_is_tokenized_by_component_not_by_word():
    profile = pb.build_profile(_app(technology_stack="Dynamics 365, Azure SQL, Power Platform"))
    tokens = profile.technical.technology_stack_tokens
    assert "dynamics 365" in tokens
    assert "azure sql" in tokens
    assert "power platform" in tokens
    assert len(tokens) == 3


def test_technology_stack_tokens_are_deduped_and_case_insensitive():
    profile = pb.build_profile(_app(technology_stack="SQL, sql, Sql"))
    assert profile.technical.technology_stack_tokens == ("sql",)


def test_missing_technology_stack_yields_empty_tokens():
    profile = pb.build_profile(_app())
    assert profile.technical.technology_stack_tokens == ()


# --- functional_redundancy_self_report: separate, never folded in ----------


def test_functional_redundancy_self_report_is_its_own_field():
    profile = pb.build_profile(_app(functional_redundancy="Partial overlap with other applications"))
    assert profile.functional_redundancy_self_report == "Partial overlap with other applications"
    # never silently folded into the functional axis
    assert not hasattr(profile.functional, "functional_redundancy")


# --- build_profiles (batch) ---------------------------------------------------


def test_build_profiles_keys_by_application_id():
    profiles = pb.build_profiles([_app("A"), _app("B")])
    assert set(profiles) == {"A", "B"}


def test_build_profiles_skips_rows_without_an_application_id():
    profiles = pb.build_profiles([{"business_capability_l1": "Finance"}, _app("B")])
    assert set(profiles) == {"B"}


# --- accepts either raw or disclosure-gated shaped dicts --------------------


def test_a_field_nulled_by_gating_is_treated_identically_to_a_blank_source_cell():
    """This module has no disclosure-classifier dependency (not an
    ancestor branch) -- it must behave identically whether None came
    from a blank cell or from gating a withheld value."""
    raw_blank = pb.build_profile(_app(business_criticality=None))
    gated_withheld = pb.build_profile(_app(business_criticality=None))  # what gating would also produce
    assert raw_blank.scale_usage.business_criticality == gated_withheld.scale_usage.business_criticality is None


# --- serialization -------------------------------------------------------------


def test_as_dict_is_json_serializable():
    profile = pb.build_profile(
        _app(
            business_capability_l1="Finance", fte_count=5, annual_fte_cost=50_000,
            technology_stack="SQL, Python",
        )
    )
    json.dumps(profile.as_dict())


# --- comparison primitives: description_similarity --------------------------


def test_description_similarity_is_jaccard_over_tokens():
    a = pb.build_profile(_app("A", application_description="Customer relationship management platform"))
    b = pb.build_profile(_app("B", application_description="Customer relationship management system"))
    similarity = pb.description_similarity(a, b)
    # shared: {customer, relationship, management} (3); union adds platform+system (5 total)
    assert similarity == round(3 / 5, 4)


def test_description_similarity_is_one_for_identical_descriptions():
    a = pb.build_profile(_app("A", application_description="Widget tracking system"))
    b = pb.build_profile(_app("B", application_description="widget tracking SYSTEM"))
    assert pb.description_similarity(a, b) == 1.0


def test_description_similarity_is_none_when_either_description_missing():
    """An absent description is not "0% similar" -- that would be a
    false, misleadingly confident signal."""
    a = pb.build_profile(_app("A", application_description="Widget tracking system"))
    b = pb.build_profile(_app("B"))
    assert pb.description_similarity(a, b) is None
    assert pb.description_similarity(b, a) is None
    assert pb.description_similarity(b, b) is None


def test_description_similarity_is_zero_for_no_meaningful_overlap():
    a = pb.build_profile(_app("A", application_description="Payroll processing engine"))
    b = pb.build_profile(_app("B", application_description="Customer support ticketing"))
    assert pb.description_similarity(a, b) == 0.0


# --- comparison primitives: capability_match_level ---------------------------


def test_capability_match_full_when_all_three_levels_match():
    a = pb.build_profile(_app("A", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"))
    b = pb.build_profile(_app("B", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_FULL


def test_capability_match_partial_when_l1_l2_match_but_l3_diverges():
    a = pb.build_profile(_app("A", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"))
    b = pb.build_profile(_app("B", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="AP"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_PARTIAL


def test_capability_match_partial_when_one_l3_is_missing():
    a = pb.build_profile(_app("A", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"))
    b = pb.build_profile(_app("B", business_capability_l1="Finance", business_capability_l2="R2R"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_PARTIAL


def test_capability_match_superficial_when_only_l1_matches():
    a = pb.build_profile(_app("A", business_capability_l1="Finance", business_capability_l2="R2R"))
    b = pb.build_profile(_app("B", business_capability_l1="Finance", business_capability_l2="Invoice to Pay"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_SUPERFICIAL


def test_capability_match_superficial_when_nothing_matches():
    a = pb.build_profile(_app("A", business_capability_l1="Finance"))
    b = pb.build_profile(_app("B", business_capability_l1="HR"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_SUPERFICIAL


def test_capability_match_is_case_insensitive():
    a = pb.build_profile(_app("A", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"))
    b = pb.build_profile(_app("B", business_capability_l1="FINANCE", business_capability_l2="r2r", business_capability_l3="gl"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_FULL


def test_capability_match_two_blank_l1_values_never_count_as_matching():
    """None == None must not be treated as a match -- two applications
    with no capability data at all are not "the same capability"."""
    a = pb.build_profile(_app("A"))
    b = pb.build_profile(_app("B"))
    assert pb.capability_match_level(a, b) == pb.CAPABILITY_MATCH_SUPERFICIAL
