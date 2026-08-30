from unittest.mock import MagicMock

import pandas as pd

from app.schemas import ApplicationInput
from app.services.agent_service import APRAgentService
from app.services.analysis_service import AnalysisService

ROW = pd.Series(
    {
        "Application ID": "APP-1",
        "Application Name": "SalesHub CRM",
        "Business Capability L1": "Customer",
        "Business Capability L2": "CRM",
        "Business Capability L3": "Sales",
        "Business Criticality": "High",
        "Strategic Relevance": "High",
        "Business Fitness": "High",
        "Usage & Adoption": "High",
        "Application Stability": "High",
        "Maintainability": "High",
        "Availability": "High",
        "Reliability": "High",
        "Scalability": "High",
        "Application Security Level": "Confidential",
        "Skill availability": "High",
        "Functional redundancy": "Low",
        "Technology Stack": "Salesforce",
        "Annual FTE Cost": 100_000.0,
        "Annual License Cost": 50_000.0,
        "FTE Count": 5,
        "Annual Infrastructure Cost": 20_000.0,
        "Other Costs": 10_000.0,
    }
)

APPLICATION_INPUT_FIXTURE = dict(
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


def test_to_scoring_input_maps_fields():
    service = AnalysisService()
    scoring_input = service._to_scoring_input(ROW)

    assert scoring_input.application_id == "APP-1"
    assert scoring_input.business_criticality == "High"
    assert scoring_input.annual_fte_cost == 100_000.0
    assert scoring_input.market_product_count == 0


def test_analyze_result_shape():
    service = AnalysisService()
    result = service.analyze(ROW)

    assert result.application_id == "APP-1"
    assert result.tim_e_score is not None
    assert result.tim_e_decision in {"Invest", "Migrate", "Tolerate", "Eliminate", "Insufficient Data"}
    assert result.cots_score is None  # batch mode never has market product retrieval
    assert "insufficient" in result.cots_recommendation.lower()


def test_analyze_withheld_cost_text_does_not_crash():
    row = ROW.copy()
    row["Annual FTE Cost"] = "cannot disclose"
    row["Annual License Cost"] = "cannot disclose"
    row["Annual Infrastructure Cost"] = "cannot disclose"
    row["Other Costs"] = "cannot disclose"

    service = AnalysisService()
    result = service.analyze(row)

    assert result.tim_e_score is not None  # qualitative axes are all still scored
    assert "withheld" in result.swot_weaknesses.lower()


def test_analyze_unrecognized_free_text_does_not_crash():
    row = ROW.copy()
    row["Business Criticality"] = "too risky"
    row["Strategic Relevance"] = "Somewhat cumbersome"
    row["Business Fitness"] = "cannot say"

    service = AnalysisService()
    result = service.analyze(row)

    assert result.tim_e_decision == "Insufficient Data"
    assert result.tim_e_score is None


def test_cross_engine_convergence():
    """CLAUDE.md section 4 bug 2: the two former scoring engines
    diverged. Pushing equivalent data through both adapters must now
    produce identical scoring results, since both call the same
    scoring kernel."""
    analysis_service = AnalysisService()
    batch_result = analysis_service.analyze(ROW)

    agent_service = APRAgentService(db=MagicMock())
    agent_service.market_service.get_products_for_application = lambda app, limit=10: []
    app_input = ApplicationInput(**APPLICATION_INPUT_FIXTURE)
    api_result = agent_service.analyze(app_input).data

    assert batch_result.tim_e_score == api_result["time_analysis"]["score"]
    assert batch_result.tim_e_decision == api_result["time_analysis"]["decision"]
    assert batch_result.cots_score == api_result["cots_analysis"]["score"]
    assert batch_result.cots_recommendation == api_result["cots_analysis"]["recommendation"]
    assert batch_result.modernization_recommendation == api_result["modernization_recommendation"]
