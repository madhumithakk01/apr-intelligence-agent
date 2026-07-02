from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class AnalysisResult:
    application_id: str
    application_name: str
    tim_e_score: float
    tim_e_decision: str
    cots_score: float
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
    """Rule-driven APR analysis engine for one application row."""

    score_map = {
        "very high": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "very low": 1,
    }

    @classmethod
    def _as_text(cls, value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    @classmethod
    def _as_number(cls, value, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _score_label(cls, value, default: int = 3) -> int:
        label = cls._as_text(value).lower()
        return cls.score_map.get(label, default)

    def analyze(self, row: pd.Series) -> AnalysisResult:
        app_id = self._as_text(row.get("Application ID", "UNKNOWN"))
        app_name = self._as_text(row.get("Application Name", "Unknown Application"))

        criticality = self._score_label(row.get("Business Criticality"))
        strategic = self._score_label(row.get("Strategic Relevance"))
        business_fitness = self._score_label(row.get("Business Fitness"))
        stability = self._score_label(row.get("Application Stability"))
        maintainability = self._score_label(row.get("Maintainability"))
        security = self._score_label(row.get("Application Security Level"))
        redundancy = self._score_label(row.get("Functional redundancy"), default=2)

        annual_fte = self._as_number(row.get("Annual FTE Cost"))
        annual_license = self._as_number(row.get("Annual License Cost"))
        annual_infra = self._as_number(row.get("Annual Infrastructure Cost"))
        other_costs = self._as_number(row.get("Other Costs"))
        total_cost = annual_fte + annual_license + annual_infra + other_costs

        cost_bucket = 1
        if total_cost >= 800000:
            cost_bucket = 5
        elif total_cost >= 500000:
            cost_bucket = 4
        elif total_cost >= 250000:
            cost_bucket = 3
        elif total_cost >= 100000:
            cost_bucket = 2

        value_score = (criticality + strategic + business_fitness) / 3
        tech_health = (stability + maintainability + security) / 3
        consolidation_need = (redundancy + cost_bucket) / 2

        tim_e_score = round((value_score * 0.45 + tech_health * 0.35 + (6 - consolidation_need) * 0.20) * 20, 2)

        if tim_e_score >= 80:
            tim_e_decision = "Invest"
        elif tim_e_score >= 60:
            tim_e_decision = "Migrate"
        elif tim_e_score >= 40:
            tim_e_decision = "Tolerate"
        else:
            tim_e_decision = "Eliminate"

        cots_score = round(((6 - redundancy) * 0.5 + (6 - maintainability) * 0.3 + (6 - stability) * 0.2) * 20, 2)
        cots_recommendation = "Replace with COTS" if cots_score >= 65 else "Retain/Enhance Current Application"

        modernization_recommendation = self._modernization_choice(tim_e_decision, tech_health, cots_score)
        rationalization_recommendation = tim_e_decision

        swot_strengths = self._strengths(row, value_score, tech_health)
        swot_weaknesses = self._weaknesses(row, redundancy, total_cost)
        swot_opportunities = self._opportunities(tim_e_decision, cots_score)
        swot_threats = self._threats(security, stability, total_cost)

        market_comparison = self._market_summary(cots_score, cots_recommendation, total_cost)
        executive_summary = self._executive_summary(
            app_name,
            tim_e_decision,
            tim_e_score,
            modernization_recommendation,
            cots_recommendation,
            total_cost,
        )

        return AnalysisResult(
            application_id=app_id,
            application_name=app_name,
            tim_e_score=tim_e_score,
            tim_e_decision=tim_e_decision,
            cots_score=cots_score,
            cots_recommendation=cots_recommendation,
            swot_strengths=swot_strengths,
            swot_weaknesses=swot_weaknesses,
            swot_opportunities=swot_opportunities,
            swot_threats=swot_threats,
            market_comparison=market_comparison,
            modernization_recommendation=modernization_recommendation,
            rationalization_recommendation=rationalization_recommendation,
            executive_summary=executive_summary,
        )

    def _modernization_choice(self, tim_e_decision: str, tech_health: float, cots_score: float) -> str:
        if tim_e_decision == "Eliminate":
            return "Retire in phased manner with business transition plan."
        if cots_score >= 70:
            return "Replace with market COTS product and migrate data."
        if tech_health < 2.5:
            return "Re-architect/refactor core components for resilience."
        if tim_e_decision == "Invest":
            return "Invest in targeted enhancements and automation."
        return "Rehost and optimize operating model."

    def _strengths(self, row: pd.Series, value_score: float, tech_health: float) -> str:
        capability = self._as_text(row.get("Business Capability L2", "core capability"))
        return (
            f"Supports {capability}; business value score is {value_score:.1f}/5; "
            f"technical health is {tech_health:.1f}/5."
        )

    def _weaknesses(self, row: pd.Series, redundancy: int, total_cost: float) -> str:
        stack = self._as_text(row.get("Technology Stack", "current stack"))
        return (
            f"Functional redundancy rating is {redundancy}/5; annual run cost is approximately "
            f"${total_cost:,.0f}; tech stack noted as {stack}."
        )

    def _opportunities(self, tim_e_decision: str, cots_score: float) -> str:
        if tim_e_decision in {"Invest", "Migrate"}:
            return "Opportunity to modernize architecture and improve integration velocity."
        if cots_score >= 65:
            return "Opportunity to adopt COTS and reduce custom maintenance footprint."
        return "Opportunity to consolidate platform footprint and optimize costs."

    def _threats(self, security: int, stability: int, total_cost: float) -> str:
        risks = []
        if security <= 2:
            risks.append("elevated security and compliance exposure")
        if stability <= 2:
            risks.append("service reliability risk")
        if total_cost >= 500000:
            risks.append("high annual run-rate pressure")
        if not risks:
            risks.append("moderate operational drift risk over time")
        return ", ".join(risks).capitalize() + "."

    def _market_summary(self, cots_score: float, cots_recommendation: str, total_cost: float) -> str:
        return (
            f"Estimated COTS suitability score: {cots_score:.2f}/100. "
            f"Recommendation: {cots_recommendation}. "
            f"Current annual total cost baseline used for comparison: ${total_cost:,.0f}."
        )

    def _executive_summary(
        self,
        app_name: str,
        tim_e_decision: str,
        tim_e_score: float,
        modernization_recommendation: str,
        cots_recommendation: str,
        total_cost: float,
    ) -> str:
        return (
            f"{app_name} is classified as {tim_e_decision} (TIM-E score {tim_e_score:.2f}/100). "
            f"Modernization recommendation: {modernization_recommendation} "
            f"COTS direction: {cots_recommendation}. "
            f"Current annual cost baseline is approximately ${total_cost:,.0f}."
        )
