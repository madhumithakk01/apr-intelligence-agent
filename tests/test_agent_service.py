from unittest.mock import MagicMock

from app.schemas import ApplicationInput
from app.services.agent_service import APRAgentService

FIXTURE = dict(
    application_id="APP-1",
    application_name="SalesHub CRM",
    owner="Jane Doe",
    owner_email="jane@example.com",
    department="Sales",
    application_description="CRM system",
    application_status="Active",
    business_criticality="High",
    business_fitness="High",
    strategic_relevance="High",
    usage_adoption="High",
    functional_redundancy="Low",
    application_security_level="Confidential",
    maintainability="High",
    application_stability="High",
    skill_availability="High",
    availability="High",
    reliability="High",
    scalability="High",
    technology_stack="Salesforce",
    annual_fte_cost=100_000.0,
    annual_license_cost=50_000.0,
    fte_count=5,
    annual_infrastructure_cost=20_000.0,
    other_costs=10_000.0,
    business_capability_l1="Customer",
    business_capability_l2="CRM",
    business_capability_l3="Sales",
)


def _make_service(monkeypatch, products):
    service = APRAgentService(db=MagicMock())
    monkeypatch.setattr(service.market_service, "get_products_for_application", lambda app, limit=10: products)
    return service


def test_to_scoring_input_maps_fields(monkeypatch):
    service = _make_service(monkeypatch, products=[])
    app_input = ApplicationInput(**FIXTURE)

    scoring_input = service._to_scoring_input(app_input, market_product_count=0)

    assert scoring_input.application_id == "APP-1"
    assert scoring_input.business_criticality == "High"
    assert scoring_input.annual_fte_cost == 100_000.0
    assert scoring_input.market_product_count == 0


def test_analyze_report_shape_no_products(monkeypatch):
    service = _make_service(monkeypatch, products=[])
    app_input = ApplicationInput(**FIXTURE)

    result = service.analyze(app_input)
    report = result.data

    for key in (
        "application",
        "time_analysis",
        "cots_analysis",
        "swot_analysis",
        "market_comparison",
        "modernization_recommendation",
        "rationalization_recommendation",
        "executive_report",
    ):
        assert key in report

    assert report["time_analysis"]["score"] is not None
    assert report["time_analysis"]["decision"] in {"Invest", "Migrate", "Tolerate", "Eliminate", "Insufficient Data"}
    assert report["cots_analysis"]["score"] is None
    assert report["cots_analysis"]["recommended_product"] is None
    assert report["market_comparison"] == []


def test_analyze_withheld_cost_does_not_crash(monkeypatch):
    service = _make_service(monkeypatch, products=[])
    fixture = dict(FIXTURE)
    fixture["annual_fte_cost"] = None
    fixture["annual_license_cost"] = None
    fixture["annual_infrastructure_cost"] = None
    fixture["other_costs"] = None
    app_input = ApplicationInput(**fixture)

    result = service.analyze(app_input)
    assert result.data["time_analysis"] is not None


def test_analyze_unrecognized_free_text_does_not_crash(monkeypatch):
    service = _make_service(monkeypatch, products=[])
    fixture = dict(FIXTURE)
    fixture["business_criticality"] = "too risky"
    fixture["strategic_relevance"] = "Somewhat cumbersome"
    fixture["business_fitness"] = "cannot say"
    app_input = ApplicationInput(**fixture)

    result = service.analyze(app_input)
    assert result.data["time_analysis"]["decision"] == "Insufficient Data"
    assert result.data["time_analysis"]["score"] is None
    assert "N/A" in result.data["executive_report"]
