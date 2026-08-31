"""Capability blocking -- SPEC.md sections 5 and 9."""

from __future__ import annotations

from app.redundancy import blocking


def _app(application_id, **overrides):
    base = {"application_id": application_id}
    base.update(overrides)
    return base


# --- primary key: (L1, L2) ---------------------------------------------


def test_apps_sharing_l1_and_l2_are_clustered_together():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("B", business_capability_l1="Finance", business_capability_l2="Record to Report"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert clusters[0].application_ids == ["A", "B"]
    assert clusters[0].blocking_tier == "l1_l2"


def test_matching_l1_l2_but_different_l3_still_cluster_together():
    """Load-bearing: this is what keeps "Partial/Component Overlap"
    (L1/L2 match, L3 diverges) reachable downstream -- see module
    docstring."""
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="GL"),
        _app("B", business_capability_l1="Finance", business_capability_l2="R2R", business_capability_l3="AP"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert set(clusters[0].application_ids) == {"A", "B"}


def test_matching_l1_but_different_l2_do_not_cluster_on_the_primary_key():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("B", business_capability_l1="Finance", business_capability_l2="Invoice to Pay"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert clusters == []


def test_key_matching_is_case_and_whitespace_insensitive():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("B", business_capability_l1="  FINANCE  ", business_capability_l2="record to report"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert set(clusters[0].application_ids) == {"A", "B"}


# --- fallback hierarchy --------------------------------------------------


def test_falls_back_to_l1_alone_when_l2_missing():
    apps = [
        _app("A", business_capability_l1="Finance"),
        _app("B", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("C", business_capability_l1="Finance"),
    ]
    clusters = blocking.block_by_capability(apps)
    # A and C share the l1_only fallback key; B has a distinct l1_l2 key
    # and is a singleton (excluded).
    assert len(clusters) == 1
    assert clusters[0].blocking_tier == "l1_only"
    assert set(clusters[0].application_ids) == {"A", "C"}


def test_falls_back_to_department_when_l1_also_missing():
    apps = [
        _app("A", department="Sales"),
        _app("B", department="Sales"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert clusters[0].blocking_tier == "department"
    assert set(clusters[0].application_ids) == {"A", "B"}


def test_never_excludes_a_row_with_no_usable_field_at_all():
    apps = [_app("A"), _app("B")]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert clusters[0].blocking_tier == "unclassified"
    assert set(clusters[0].application_ids) == {"A", "B"}


def test_blank_strings_are_treated_the_same_as_missing():
    apps = [
        _app("A", business_capability_l1="   ", department="Sales"),
        _app("B", department="Sales"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 1
    assert clusters[0].blocking_tier == "department"


def test_a_more_specific_tier_never_falls_back_even_if_a_coarser_group_exists():
    """An app with full L1+L2 must not also get swept into an L1-only or
    Department bucket -- one row belongs to exactly one cluster."""
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report", department="Finance Ops"),
        _app("B", business_capability_l1="Finance", business_capability_l2="Record to Report", department="Finance Ops"),
        _app("C", business_capability_l1="Finance", department="Finance Ops"),
        _app("D", business_capability_l1="Finance", department="Finance Ops"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert len(clusters) == 2
    tiers = {c.blocking_tier: set(c.application_ids) for c in clusters}
    assert tiers["l1_l2"] == {"A", "B"}
    assert tiers["l1_only"] == {"C", "D"}


# --- singletons and structure --------------------------------------------


def test_singleton_clusters_are_excluded():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("B", business_capability_l1="HR", business_capability_l2="Hire to Retire"),
    ]
    assert blocking.block_by_capability(apps) == []


def test_empty_portfolio_yields_no_clusters():
    assert blocking.block_by_capability([]) == []


def test_a_row_without_an_application_id_is_skipped():
    apps = [
        {"business_capability_l1": "Finance", "business_capability_l2": "R2R"},
        _app("B", business_capability_l1="Finance", business_capability_l2="R2R"),
    ]
    assert blocking.block_by_capability(apps) == []


def test_cluster_ids_are_deterministic_and_readable():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="Record to Report"),
        _app("B", business_capability_l1="Finance", business_capability_l2="Record to Report"),
    ]
    clusters = blocking.block_by_capability(apps)
    assert clusters[0].cluster_id == "CL-L1L2-FINANCE-RECORD-TO-REPORT"


def test_rerunning_blocking_on_the_same_input_is_stable():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="R2R"),
        _app("B", business_capability_l1="Finance", business_capability_l2="R2R"),
        _app("C", business_capability_l1="HR", business_capability_l2="H2R"),
        _app("D", business_capability_l1="HR", business_capability_l2="H2R"),
    ]
    first = [c.as_dict() for c in blocking.block_by_capability(apps)]
    second = [c.as_dict() for c in blocking.block_by_capability(apps)]
    assert first == second


def test_as_dict_is_json_serializable_and_orchestration_shaped():
    """SPEC.md section 13's target orchestration graph consumes
    clusters as {"cluster_id", "application_ids"} -- as_dict() must be a
    superset of that shape."""
    import json

    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="R2R"),
        _app("B", business_capability_l1="Finance", business_capability_l2="R2R"),
    ]
    payload = blocking.block_by_capability(apps)[0].as_dict()
    json.dumps(payload)
    assert set(payload) >= {"cluster_id", "application_ids"}
    assert payload["application_ids"] == ["A", "B"]


def test_multiple_distinct_clusters_all_present():
    apps = [
        _app("A", business_capability_l1="Finance", business_capability_l2="R2R"),
        _app("B", business_capability_l1="Finance", business_capability_l2="R2R"),
        _app("C", business_capability_l1="HR", business_capability_l2="H2R"),
        _app("D", business_capability_l1="HR", business_capability_l2="H2R"),
        _app("E", business_capability_l1="HR", business_capability_l2="H2R"),
    ]
    clusters = blocking.block_by_capability(apps)
    sizes = sorted(len(c.application_ids) for c in clusters)
    assert sizes == [2, 3]
