from dataclasses import dataclass
from typing import Dict
from typing import List

from sqlalchemy.orm import Session

from app.schemas import ApplicationInput
from app.services.market_service import MarketService
from app.services.market_service import StructuredMarketProduct


@dataclass
class AgentResult:
    data: Dict


class APRAgentService:
    score_map = {
        "very high": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "very low": 1,
    }

    def __init__(self, db: Session):
        self.db = db
        self.market_service = MarketService(db)

    def _score(self, value: str, default: int = 3) -> int:
        return self.score_map.get((value or "").strip().lower(), default)

    def _cost_bucket(self, total_cost: float) -> int:
        if total_cost >= 800000:
            return 5
        if total_cost >= 500000:
            return 4
        if total_cost >= 250000:
            return 3
        if total_cost >= 100000:
            return 2
        return 1

    def _tim_e(self, app: ApplicationInput) -> Dict:
        criticality = self._score(app.business_criticality)
        strategic = self._score(app.strategic_relevance)
        fitness = self._score(app.business_fitness)
        stability = self._score(app.application_stability)
        maintainability = self._score(app.maintainability)
        security = self._score(app.application_security_level)
        redundancy = self._score(app.functional_redundancy, default=2)

        total_cost = (
            app.annual_fte_cost
            + app.annual_license_cost
            + app.annual_infrastructure_cost
            + app.other_costs
        )
        value_score = (criticality + strategic + fitness) / 3
        health_score = (stability + maintainability + security) / 3
        consolidation_need = (redundancy + self._cost_bucket(total_cost)) / 2

        tim_e_score = round(
            (value_score * 0.45 + health_score * 0.35 + (6 - consolidation_need) * 0.20) * 20,
            2,
        )
        if tim_e_score >= 80:
            decision = "Invest"
        elif tim_e_score >= 60:
            decision = "Migrate"
        elif tim_e_score >= 40:
            decision = "Tolerate"
        else:
            decision = "Eliminate"
        return {
            "score": tim_e_score,
            "decision": decision,
            "value_score": value_score,
            "health_score": health_score,
        }

    def _cots_fit(self, app: ApplicationInput, products: List[StructuredMarketProduct]) -> Dict:
        if not products:
            return {
                "score": 0.0,
                "recommendation": "Retain existing application — insufficient retrieved market data for COTS comparison",
                "recommended_product": None,
            }

        redundancy = self._score(app.functional_redundancy, default=2)
        maintainability = self._score(app.maintainability)
        stability = self._score(app.application_stability)
        base_score = ((6 - redundancy) * 0.5 + (6 - maintainability) * 0.3 + (6 - stability) * 0.2) * 20
        cots_score = round(min(base_score + min(len(products), 10) * 1.5, 100), 2)
        recommendation = "Replace with COTS" if cots_score >= 65 else "Retain/Enhance Existing Application"
        recommended = products[0].product_name if cots_score >= 65 else None
        return {
            "score": cots_score,
            "recommendation": recommendation,
            "recommended_product": recommended,
        }

    def _swot(
        self,
        app: ApplicationInput,
        products: List[StructuredMarketProduct],
        tim_e: Dict,
        cots: Dict,
    ) -> Dict:
        strengths = (
            f"{app.application_name} has business alignment score {tim_e['value_score']:.1f}/5 and supports "
            f"{app.business_capability_l2} ({app.business_capability_l3})."
        )
        weaknesses = (
            f"Technical health score is {tim_e['health_score']:.1f}/5 with redundancy level "
            f"{app.functional_redundancy}. Maintainability is rated {app.maintainability}."
        )

        if not products:
            return {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "opportunities": "Market product data was not retrieved; opportunity assessment deferred until COTS research completes.",
                "threats": "Unable to benchmark against current market products without retrieved COTS intelligence.",
                "comparison_basis": "No retrieved market products available for comparison in this run.",
                "cots_context": cots["recommendation"],
            }

        product_names = [p.product_name for p in products[:5]]
        top = products[0]
        cloud_count = sum(1 for p in products if "cloud" in p.deployment_model.lower() or "saas" in p.deployment_model.lower())
        ai_count = sum(1 for p in products if any("not mentioned" not in f.lower() for f in p.ai_features))

        opportunities = (
            f"Retrieved alternatives ({', '.join(product_names)}) offer modernization paths. "
            f"{top.product_name} ({top.vendor}) supports {top.deployment_model} deployment with "
            f"licensing model: {top.licensing_model}."
        )
        threats = (
            f"{cloud_count} of {len(products)} retrieved products are cloud/SaaS oriented; "
            f"{ai_count} mention AI capabilities. Delayed action may widen feature and cost gaps versus "
            f"{', '.join(product_names[:3])}."
        )
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats,
            "comparison_basis": f"SWOT compared against {len(products)} Tavily-retrieved enterprise products only.",
            "cots_context": cots["recommendation"],
        }

    def _modernization(self, tim_e_decision: str, cots: Dict) -> str:
        if tim_e_decision == "Eliminate":
            return "Retire over a phased 2-quarter transition plan."
        if cots.get("recommended_product"):
            return (
                f"Evaluate {cots['recommended_product']} via fit-gap workshop, POC, and phased data migration."
            )
        if cots["recommendation"].startswith("Retain existing"):
            return "Retain and monitor; complete targeted COTS research before replacement decision."
        if tim_e_decision == "Invest":
            return "Invest in refactoring, API hardening, and observability."
        if tim_e_decision == "Migrate":
            return "Migrate to cloud-native architecture with minimal downtime."
        return "Tolerate short-term while preparing modernization backlog."

    def analyze(self, app: ApplicationInput) -> AgentResult:
        products = self.market_service.get_products_for_application(app, limit=10)
        tim_e = self._tim_e(app)
        cots = self._cots_fit(app, products)
        swot = self._swot(app, products, tim_e, cots)
        modernization = self._modernization(tim_e["decision"], cots)

        market_comparison = [product.to_dict() for product in products]

        executive_parts = [
            f"{app.application_name} is categorized as {tim_e['decision']} with TIM-E score {tim_e['score']}.",
            f"COTS recommendation: {cots['recommendation']}.",
        ]
        if cots.get("recommended_product"):
            executive_parts.append(f"Primary COTS candidate from retrieved data: {cots['recommended_product']}.")
        executive_parts.append(f"Suggested action: {modernization}")
        if products:
            executive_parts.append(
                f"Market intelligence: {len(products)} enterprise products retrieved via Tavily for {app.business_capability_l2}."
            )
        else:
            executive_parts.append("Market intelligence: no products retrieved in this run.")

        report_data = {
            "application": app.model_dump(),
            "time_analysis": tim_e,
            "cots_analysis": cots,
            "swot_analysis": swot,
            "market_comparison": market_comparison,
            "modernization_recommendation": modernization,
            "rationalization_recommendation": tim_e["decision"],
            "executive_report": " ".join(executive_parts),
        }
        return AgentResult(data=report_data)
