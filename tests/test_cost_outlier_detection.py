"""Cost outlier detection -- SPEC.md sections 5, 9, 12. Purely
deterministic: no LLM mocking needed anywhere in this file."""

from __future__ import annotations

from app.cost_intelligence import outlier_detection as od
from app.redundancy.profile_builder import build_profile


def _profile(application_id, fte_count, annual_fte_cost):
    return build_profile(
        {
            "application_id": application_id,
            "fte_count": fte_count,
            "annual_fte_cost": annual_fte_cost,
            "annual_license_cost": 0,
            "annual_infrastructure_cost": 0,
            "other_costs": 0,
        }
    )


# --- the peer-count floor ----------------------------------------------


def test_fewer_than_five_known_peers_never_flags_anything():
    # Tight enough (IQR ~= 0) that the statistics alone would flag A3 if
    # the floor didn't suppress it outright -- see
    # test_exactly_five_known_peers_is_the_floor_not_excluded for
    # confirmation that 5 of these same peers *do* get evaluated.
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_000, 10_000, 10_001])]
    flags = od.detect_cluster_outliers("CL-1", profiles)
    assert flags == []


def test_exactly_five_known_peers_is_the_floor_not_excluded():
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 11_000, 12_000, 13_000, 1_000_000])]
    flags = od.detect_cluster_outliers("CL-1", profiles)
    assert len(flags) == 1
    assert flags[0].application_id == "A4"


def test_peers_with_unknown_cost_do_not_count_toward_the_floor():
    """5 known + 3 unknown-cost peers -- the unknowns must not pad the
    count past the floor on their own."""
    known = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 11_000, 12_000, 13_000, 14_000])]
    unknown = [build_profile({"application_id": f"U{i}"}) for i in range(3)]  # no cost fields at all
    flags = od.detect_cluster_outliers("CL-1", known + unknown)
    assert flags == []  # 5 known peers, all tightly clustered -- no outlier, and unknowns never flagged


def test_a_member_with_unknown_cost_is_never_flagged():
    known = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 11_000, 12_000, 13_000, 1_000_000])]
    unknown = build_profile({"application_id": "U1"})
    flags = od.detect_cluster_outliers("CL-1", known + [unknown])
    assert all(flag.application_id != "U1" for flag in flags)


# --- IQR fence direction and correctness ------------------------------


def test_flags_a_high_outlier():
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_500, 11_000, 11_500, 500_000])]
    flags = od.detect_cluster_outliers("CL-1", profiles)
    assert len(flags) == 1
    assert flags[0].direction == od.DIRECTION_HIGH
    assert flags[0].application_id == "A4"


def test_flags_a_low_outlier():
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([100_000, 101_000, 102_000, 103_000, 500])]
    flags = od.detect_cluster_outliers("CL-1", profiles)
    assert len(flags) == 1
    assert flags[0].direction == od.DIRECTION_LOW
    assert flags[0].application_id == "A4"


def test_a_tight_cluster_with_no_deviation_flags_nothing():
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 9_950])]
    assert od.detect_cluster_outliers("CL-1", profiles) == []


def test_multiple_outliers_in_the_same_cluster_are_all_flagged():
    profiles = [
        _profile(f"A{i}", 1, cost)
        for i, cost in enumerate([10_000, 10_500, 11_000, 11_500, 12_000, 1_000_000, 100])
    ]
    flags = od.detect_cluster_outliers("CL-1", profiles)
    flagged_ids = {flag.application_id for flag in flags}
    assert flagged_ids == {"A5", "A6"}


def test_cluster_stats_reflect_only_the_known_peer_population():
    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 11_000, 12_000, 13_000, 1_000_000])]
    unknown = build_profile({"application_id": "U1"})
    flags = od.detect_cluster_outliers("CL-1", profiles + [unknown])
    assert flags[0].cluster_stats.peer_count == 5


# --- cost-per-fte, never a raw total ------------------------------------


def test_uses_normalized_cost_per_fte_not_raw_total():
    """Two applications with the same raw total cost but very different
    FTE counts must not be treated as having the same cost signal."""
    small_team = _profile("SMALL", 1, 100_000)  # cost-per-fte = 100,000
    large_team = build_profile(
        {"application_id": "LARGE", "fte_count": 100, "annual_fte_cost": 100_000,
         "annual_license_cost": 0, "annual_infrastructure_cost": 0, "other_costs": 0}
    )  # cost-per-fte = 1,000
    peers = [_profile(f"P{i}", 1, 100_000) for i in range(4)]  # same cost-per-fte as small_team
    flags = od.detect_cluster_outliers("CL-1", peers + [small_team, large_team])
    flagged_ids = {flag.application_id for flag in flags}
    assert "LARGE" in flagged_ids  # far below the tight cost-per-fte cluster
    assert "SMALL" not in flagged_ids  # matches the peer cost-per-fte exactly


# --- detect_cost_outliers: cluster-dict driven batch entry point -------


def test_detect_cost_outliers_iterates_every_cluster():
    clusters = [
        {"cluster_id": "CL-A", "application_ids": [f"A{i}" for i in range(5)]},
        {"cluster_id": "CL-B", "application_ids": [f"B{i}" for i in range(5)]},
    ]
    profiles = {
        **{f"A{i}": _profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 500_000])},
        **{f"B{i}": _profile(f"B{i}", 1, cost) for i, cost in enumerate([1_000, 1_010, 990, 1_005, 995])},
    }
    flags = od.detect_cost_outliers(clusters, profiles)
    assert {flag.application_id for flag in flags} == {"A4"}
    assert flags[0].cluster_id == "CL-A"


def test_detect_cost_outliers_skips_a_cluster_below_the_floor():
    clusters = [{"cluster_id": "CL-A", "application_ids": ["A0", "A1", "A2"]}]
    profiles = {f"A{i}": _profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_000, 1_000_000])}
    assert od.detect_cost_outliers(clusters, profiles) == []


def test_detect_cost_outliers_ignores_an_application_id_missing_from_profiles():
    clusters = [{"cluster_id": "CL-A", "application_ids": [f"A{i}" for i in range(5)] + ["MISSING"]}]
    profiles = {f"A{i}": _profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 500_000])}
    flags = od.detect_cost_outliers(clusters, profiles)
    assert all(flag.application_id != "MISSING" for flag in flags)


def test_empty_clusters_list_flags_nothing():
    assert od.detect_cost_outliers([], {}) == []


# --- serialization ------------------------------------------------------


def test_as_dict_is_json_serializable():
    import json

    profiles = [_profile(f"A{i}", 1, cost) for i, cost in enumerate([10_000, 10_100, 9_900, 10_050, 500_000])]
    flag = od.detect_cluster_outliers("CL-1", profiles)[0]
    json.dumps(flag.as_dict())
