from dataclasses import dataclass
from typing import Dict
from typing import Optional

import pandas as pd

from app.ingestion.cost_parsing import parse_cost_cell
from app.scoring.kernel import ScoringInput
from app.scoring.kernel import score_application


def _fmt_score(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def _fmt_cost(value: Optional[float]) -> str:
    return f"${value:,.0f}" if value is not None else "withheld/unavailable"


@dataclass
class AnalysisResult:
    application_id: str
    application_name: str
    tim_e_score: Optional[float]
    tim_e_decision: str
    cots_score: Optional[float]
    cots_recommendation: str
    swot_strengths: str
    swot_weaknesses: str
    swot_opportunities: str
    swot_threats: str
    market_comparison: str
    modernization_recommendation: str
    rationalization_recommendation: str
    executive_summary: str

    def to_dict(self) -> Dict[str, str]:
        return self.__dict__


class AnalysisService:
    """Rule-driven APR analysis engine for one application row.

    Batch mode (this class, via run_apr_prototype.py) never performs
    market-product retrieval -- unlike the interactive API path
    (app/services/agent_service.py), it has no MarketService/Tavily
    integration. Consolidating both engines onto one scoring kernel
    (scoring.kernel.compute_cots_fit) means a COTS recommendation here
    is now honestly reported as "insufficient retrieved market data"
    for every row, rather than the old batch engine's prior behavior of
    computing a COTS-fit number from qualitative axes alone with no
    market evidence backing the claim at all.
    """

    @classmethod
    def _as_text(cls, value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _to_scoring_input(self, row: pd.Series) -> ScoringInput:
        application_id = self._as_text(row.get("Application ID", "UNKNOWN"))
        cost_fields = {
            "annual_fte_cost": "Annual FTE Cost",
            "annual_license_cost": "Annual License Cost",
            "annual_infrastructure_cost": "Annual Infrastructure Cost",
            "other_costs": "Other Costs",
        }
        parsed_costs = {
            field_name: parse_cost_cell(row.get(column), field_name=field_name, application_id=application_id)
            for field_name, column in cost_fields.items()
        }

        return ScoringInput(
            application_id=application_id,
            application_name=self._as_text(row.get("Application Name", "Unknown Application")),
            business_capability_l2=self._as_text(row.get("Business Capability L2")),
            business_capability_l3=self._as_text(row.get("Business Capability L3")),
            business_criticality=self._as_text(row.get("Business Criticality")),
            strategic_relevance=self._as_text(row.get("Strategic Relevance")),
            business_fitness=self._as_text(row.get("Business Fitness")),
            usage_adoption=self._as_text(row.get("Usage & Adoption")),
            application_stability=self._as_text(row.get("Application Stability")),
            maintainability=self._as_text(row.get("Maintainability")),
            availability=self._as_text(row.get("Availability")),
            reliability=self._as_text(row.get("Reliability")),
            scalability=self._as_text(row.get("Scalability")),
            application_security_level=self._as_text(row.get("Application Security Level")),
            skill_availability=self._as_text(row.get("Skill availability")),
            functional_redundancy=self._as_text(row.get("Functional redundancy")),
            annual_fte_cost=parsed_costs["annual_fte_cost"].value,
            annual_license_cost=parsed_costs["annual_license_cost"].value,
            annual_infrastructure_cost=parsed_costs["annual_infrastructure_cost"].value,
            other_costs=parsed_costs["other_costs"].value,
            market_product_count=0,
        )

    @staticmethod
    def _total_known_cost(scoring_input: ScoringInput) -> Optional[float]:
        components = [
            scoring_input.annual_fte_cost,
            scoring_input.annual_license_cost,
            scoring_input.annual_infrastructure_cost,
            scoring_input.other_costs,
        ]
        known = [c for c in components if c is not None]
        return sum(known) if known else None

    def analyze(self, row: pd.Series) -> AnalysisResult:
        scoring_input = self._to_scoring_input(row)
        result = score_application(scoring_input)
        total_cost = self._total_known_cost(scoring_input)

        swot_strengths = self._strengths(row, result.tim_e.value_score.value, result.tim_e.health_score.value)
        swot_weaknesses = self._weaknesses(row, scoring_input.functional_redundancy, total_cost)
        swot_opportunities = self._opportunities(result.tim_e.decision, result.cots.meets_threshold)
        swot_threats = self._threats(
            result.tim_e.security_classification,
            self._as_text(row.get("Application Stability")),
            total_cost,
        )

        market_comparison = self._market_summary(result.cots.score, result.cots.recommendation, total_cost)
        executive_summary = self._executive_summary(
            scoring_input.application_name,
            result.tim_e.decision,
            result.tim_e.score,
            result.modernization_recommendation,
            result.cots.recommendation,
            total_cost,
        )

        return AnalysisResult(
            application_id=scoring_input.application_id,
            application_name=scoring_input.application_name,
            tim_e_score=result.tim_e.score,
            tim_e_decision=result.tim_e.decision,
            cots_score=result.cots.score,
            cots_recommendation=result.cots.recommendation,
            swot_strengths=swot_strengths,
            swot_weaknesses=swot_weaknesses,
            swot_opportunities=swot_opportunities,
            swot_threats=swot_threats,
            market_comparison=market_comparison,
            modernization_recommendation=result.modernization_recommendation,
            rationalization_recommendation=result.tim_e.decision,
            executive_summary=executive_summary,
        )

    def _strengths(self, row: pd.Series, value_score: Optional[float], tech_health: Optional[float]) -> str:
        capability = self._as_text(row.get("Business Capability L2")) or "core capability"
        return (
            f"Supports {capability}; business value score is {_fmt_score(value_score)}/5; "
            f"technical health is {_fmt_score(tech_health)}/5."
        )

    def _weaknesses(self, row: pd.Series, functional_redundancy: str, total_cost: Optional[float]) -> str:
        stack = self._as_text(row.get("Technology Stack")) or "current stack"
        redundancy_label = functional_redundancy or "not disclosed"
        return (
            f"Functional redundancy rating is {redundancy_label}; annual run cost is approximately "
            f"{_fmt_cost(total_cost)}; tech stack noted as {stack}."
        )

    def _opportunities(self, tim_e_decision: str, meets_cots_threshold: bool) -> str:
        if tim_e_decision in {"Invest", "Migrate"}:
            return "Opportunity to modernize architecture and improve integration velocity."
        if meets_cots_threshold:
            return "Opportunity to adopt COTS and reduce custom maintenance footprint."
        return "Opportunity to consolidate platform footprint and optimize costs."

    def _threats(
        self,
        security_classification: Optional[str],
        stability_label: str,
        total_cost: Optional[float],
    ) -> str:
        risks = []
        if security_classification:
            risks.append(f"data classification noted as {security_classification}")
        if stability_label.strip().lower() in {"low", "very low"}:
            risks.append("service reliability risk")
        if total_cost is not None and total_cost >= 500000:
            risks.append("high annual run-rate pressure")
        if not risks:
            risks.append("moderate operational drift risk over time")
        return ", ".join(risks).capitalize() + "."

    def _market_summary(self, cots_score: Optional[float], cots_recommendation: str, total_cost: Optional[float]) -> str:
        score_text = f"{cots_score:.2f}/100" if cots_score is not None else "not computed (no market data retrieved)"
        return (
            f"Estimated COTS suitability score: {score_text}. "
            f"Recommendation: {cots_recommendation}. "
            f"Current annual total cost baseline used for comparison: {_fmt_cost(total_cost)}."
        )

    def _executive_summary(
        self,
        app_name: str,
        tim_e_decision: str,
        tim_e_score: Optional[float],
        modernization_recommendation: str,
        cots_recommendation: str,
        total_cost: Optional[float],
    ) -> str:
        return (
            f"{app_name} is classified as {tim_e_decision} (TIM-E score {_fmt_score(tim_e_score)}/100). "
            f"Modernization recommendation: {modernization_recommendation} "
            f"COTS direction: {cots_recommendation}. "
            f"Current annual cost baseline is approximately {_fmt_cost(total_cost)}."
        )
