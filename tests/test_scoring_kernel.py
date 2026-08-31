import ast
from pathlib import Path

import pytest

from app.scoring import governance_params as gp
from app.scoring import kernel


def _base_kwargs(**overrides):
    kwargs = dict(
        application_id="APP-1",
        application_name="Test App",
        business_capability_l2="CRM",
        business_capability_l3="Sales",
        business_criticality="High",
        strategic_relevance="High",
        business_fitness="High",
        usage_adoption="High",
        application_stability="High",
        maintainability="High",
        availability="High",
        reliability="High",
        scalability="High",
        application_security_level="Confidential",
        skill_availability="High",
        functional_redundancy="Low",
        annual_fte_cost=100_000.0,
        annual_license_cost=50_000.0,
        annual_infrastructure_cost=20_000.0,
        other_costs=10_000.0,
        market_product_count=0,
    )
    kwargs.update(overrides)
    return kwargs


def _input(**overrides) -> kernel.ScoringInput:
    return kernel.ScoringInput(**_base_kwargs(**overrides))


@pytest.mark.parametrize(
    "label,expected",
    [
        ("very high", 5),
        ("High", 4),
        ("  medium  ", 3),
        ("Low", 2),
        ("very low", 1),
        ("VERY HIGH", 5),
    ],
)
def test_known_labels_score_correctly(label, expected):
    assert kernel.score_qualitative_label(label) == expected


@pytest.mark.parametrize("value", ["too risky", "Somewhat cumbersome", "cannot say", None, "", "   "])
def test_unrecognized_free_text_is_unscored(value):
    assert kernel.score_qualitative_label(value) is None


def test_insufficient_data_when_too_many_axes_unscored():
    inputs = _input(
        business_criticality="cannot say",
        strategic_relevance="too risky",
        business_fitness=None,
        usage_adoption="High",
    )
    result = kernel.compute_tim_e(inputs)
    assert result.decision == "Insufficient Data"
    assert result.score is None
    assert result.value_score.status == "insufficient_data"


def test_security_level_excluded_from_health_score():
    confidential = _input(application_security_level="Confidential")
    internal = _input(application_security_level="Internal use only")

    result_confidential = kernel.compute_tim_e(confidential)
    result_internal = kernel.compute_tim_e(internal)

    assert result_confidential.health_score.value == result_internal.health_score.value
    assert result_confidential.security_classification == "Confidential"
    assert result_internal.security_classification == "Internal use only"
    assert "application_security_level" not in result_confidential.health_score.scored_axes
    assert "application_security_level" not in result_confidential.health_score.unscored_axes


@pytest.mark.parametrize("field", ["availability", "reliability", "scalability"])
def test_dead_fields_move_health_score(field):
    baseline = kernel.compute_tim_e(_input()).health_score.value
    lowered = kernel.compute_tim_e(_input(**{field: "very low"})).health_score.value
    assert lowered < baseline


def test_usage_adoption_moves_value_score():
    baseline = kernel.compute_tim_e(_input()).value_score.value
    lowered = kernel.compute_tim_e(_input(usage_adoption="very low")).value_score.value
    assert lowered < baseline


def _high_score_kwargs(**overrides):
    # Every axis maxed out and cost minimized so the raw weighted score
    # clears the Invest threshold on its own -- lets a test isolate
    # whether a later override (e.g. Low stability) triggers the floor
    # versus just dragging the raw arithmetic down instead.
    kwargs = dict(
        business_criticality="very high",
        strategic_relevance="very high",
        business_fitness="very high",
        usage_adoption="very high",
        application_stability="very high",
        maintainability="very high",
        availability="very high",
        reliability="very high",
        scalability="very high",
        functional_redundancy="very low",
        annual_fte_cost=1_000.0,
        annual_license_cost=1_000.0,
        annual_infrastructure_cost=1_000.0,
        other_costs=1_000.0,
        skill_availability="very high",
    )
    kwargs.update(overrides)
    return kwargs


def test_skill_availability_floor_overrides_invest():
    inputs = _input(**_high_score_kwargs(application_stability="Low", skill_availability="Low"))
    result = kernel.compute_tim_e(inputs)
    assert result.raw_decision == "Invest"
    assert result.decision == "Migrate"
    assert result.floor_applied == "skill_availability_floor"

    # the floor changes the decision only, never the underlying number
    unfloored_inputs = _input(**_high_score_kwargs(application_stability="Low"))
    unfloored = kernel.compute_tim_e(unfloored_inputs)
    assert result.score == unfloored.score


def test_floor_not_applied_when_stability_unscored():
    inputs = _input(skill_availability="Low", application_stability=None)
    result = kernel.compute_tim_e(inputs)
    assert result.floor_applied is None
    assert result.decision == result.raw_decision


def test_floor_not_applied_when_decision_not_invest():
    inputs = _input(
        skill_availability="Low",
        application_stability="Low",
        business_criticality="Low",
        strategic_relevance="Low",
        business_fitness="Low",
        usage_adoption="Low",
    )
    result = kernel.compute_tim_e(inputs)
    assert result.raw_decision != "Invest"
    assert result.floor_applied is None
    assert result.decision == result.raw_decision


def test_cots_threshold_is_65():
    assert gp.COTS_REPLACE_THRESHOLD == 65


def test_cots_threshold_boundary_is_inclusive(monkeypatch):
    inputs = _input(market_product_count=1)
    baseline = kernel.compute_cots_fit(inputs)
    assert baseline.score is not None
    score = baseline.score

    monkeypatch.setattr(gp, "COTS_REPLACE_THRESHOLD", score)
    at_threshold = kernel.compute_cots_fit(inputs)
    assert at_threshold.meets_threshold is True
    assert at_threshold.recommendation == "Replace with COTS"

    monkeypatch.setattr(gp, "COTS_REPLACE_THRESHOLD", score + 0.01)
    above_score = kernel.compute_cots_fit(inputs)
    assert above_score.meets_threshold is False
    assert above_score.recommendation == "Retain/Enhance Existing Application"


def test_cots_no_products_means_no_score():
    result = kernel.compute_cots_fit(_input(market_product_count=0))
    assert result.score is None
    assert result.meets_threshold is False
    assert "insufficient" in result.recommendation.lower()


def _numeric_65_or_70_literal_lines(path: Path):
    """AST-based, not text-based: only real numeric literals in code can
    match (a docstring/comment mentioning "70" is a str constant, never
    an int/float one, so prose can never produce a false positive)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value in (65, 70)
    ]


def test_no_second_cots_threshold_in_source():
    """CLAUDE.md section 4 bug 3: one canonical COTS threshold, read
    from governance_params. The two former scoring engines that held the
    conflicting 65-vs-70 literals (agent_service.py, analysis_service.py)
    were deleted once the async pipeline replaced them, so kernel.py is
    the only engine left to guard."""
    kernel_path = Path(__file__).resolve().parent.parent / "app" / "scoring" / "kernel.py"
    lines = _numeric_65_or_70_literal_lines(kernel_path)
    assert lines == [], f"Bare 65/70 numeric literal found in kernel.py at line(s) {lines}"


def test_withheld_cost_does_not_crash_and_is_not_zeroed():
    inputs = _input(
        annual_fte_cost=None,
        annual_license_cost=None,
        annual_infrastructure_cost=None,
        other_costs=None,
    )
    result = kernel.compute_tim_e(inputs)
    # cost_bucket is None -> consolidation_need falls back to redundancy alone
    assert "cost_bucket" in result.consolidation_need.unscored_axes


def test_partial_cost_flagged_incomplete():
    cost = kernel._aggregate_cost(100_000.0, None, 20_000.0, None)
    assert cost.is_complete is False
    assert cost.total_known == 120_000.0
    assert set(cost.missing_fields) == {"annual_license_cost", "other_costs"}


def test_all_cost_missing_has_none_total():
    cost = kernel._aggregate_cost(None, None, None, None)
    assert cost.total_known is None
    assert cost.is_complete is False


def test_governance_params_values_match_claude_md():
    assert gp.TIME_WEIGHTS == {"value": 0.45, "health": 0.35, "consolidation": 0.20}
    assert gp.DECISION_THRESHOLDS == {"invest": 80, "migrate": 60, "tolerate": 40}
    assert gp.COTS_REPLACE_THRESHOLD == 65
    assert gp.QUALITATIVE_ENSEMBLE_SIZE == 3
    assert gp.REDUNDANCY_ENSEMBLE_SIZE == 3
    assert gp.MARKET_AGENT_ITERATION_CAP == 4
    assert gp.MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER == 5
    # QUALITATIVE_ESCALATION_CONFIDENCE_THRESHOLD: set by feat/qualitative-scoring
    # (branch 8), not itemized in CLAUDE.md section 12's table as a fixed value --
    # see tests/test_qualitative_scoring.py for its own governance-params coverage.


def test_score_application_end_to_end():
    inputs = _input(market_product_count=3)
    result = kernel.score_application(inputs)
    assert result.tim_e.score is not None
    assert result.modernization_recommendation
