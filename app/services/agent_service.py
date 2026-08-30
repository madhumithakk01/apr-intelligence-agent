from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from sqlalchemy.orm import Session

from app.schemas import ApplicationInput
from app.scoring.kernel import ScoringInput
from app.scoring.kernel import score_application
from app.services.market_service import MarketService
from app.services.market_service import StructuredMarketProduct


@dataclass
class AgentResult:
    data: Dict


def _fmt_score(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


class APRAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.market_service = MarketService(db)

    def _to_scoring_input(self, app: ApplicationInput, market_product_count: int) -> ScoringInput:
        return ScoringInput(
            application_id=app.application_id,
            application_name=app.application_name,
            business_capability_l2=app.business_capability_l2,
            business_capability_l3=app.business_capability_l3,
            business_criticality=app.business_criticality,
            strategic_relevance=app.strategic_relevance,
            business_fitness=app.business_fitness,
            usage_adoption=app.usage_adoption,
            application_stability=app.application_stability,
            maintainability=app.maintainability,
            availability=app.availability,
            reliability=app.reliability,
            scalability=app.scalability,
            application_security_level=app.application_security_level,
            skill_availability=app.skill_availability,
            functional_redundancy=app.functional_redundancy,
            annual_fte_cost=app.annual_fte_cost,
            annual_license_cost=app.annual_license_cost,
            annual_infrastructure_cost=app.annual_infrastructure_cost,
            other_costs=app.other_costs,
            market_product_count=market_product_count,
        )

    def _swot(
        self,
        app: ApplicationInput,
        products: List[StructuredMarketProduct],
        time_analysis: Dict,
        cots_analysis: Dict,
    ) -> Dict:
        strengths = (
            f"{app.application_name} has business alignment score {_fmt_score(time_analysis['value_score'])}/5 "
            f"and supports {app.business_capability_l2} ({app.business_capability_l3})."
        )
        weaknesses = (
            f"Technical health score is {_fmt_score(time_analysis['health_score'])}/5 with redundancy level "
            f"{app.functional_redundancy}. Maintainability is rated {app.maintainability}."
        )

        if not products:
            return {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "opportunities": "Market product data was not retrieved; opportunity assessment deferred until COTS research completes.",
                "threats": "Unable to benchmark against current market products without retrieved COTS intelligence.",
                "comparison_basis": "No retrieved market products available for comparison in this run.",
                "cots_context": cots_analysis["recommendation"],
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
            "cots_context": cots_analysis["recommendation"],
        }

    def analyze(self, app: ApplicationInput) -> AgentResult:
        products = self.market_service.get_products_for_application(app, limit=10)

        scoring_input = self._to_scoring_input(app, market_product_count=len(products))
        result = score_application(scoring_input)

        time_analysis = {
            "score": result.tim_e.score,
            "decision": result.tim_e.decision,
            "value_score": result.tim_e.value_score.value,
            "health_score": result.tim_e.health_score.value,
        }
        cots_analysis = {
            "score": result.cots.score,
            "recommendation": result.cots.recommendation,
            "recommended_product": products[0].product_name if result.cots.meets_threshold and products else None,
        }
        modernization = result.modernization_recommendation

        swot = self._swot(app, products, time_analysis, cots_analysis)

        market_comparison = [product.to_dict() for product in products]

        executive_parts = [
            f"{app.application_name} is categorized as {time_analysis['decision']} with TIM-E score "
            f"{_fmt_score(time_analysis['score'])}.",
            f"COTS recommendation: {cots_analysis['recommendation']}.",
        ]
        if cots_analysis.get("recommended_product"):
            executive_parts.append(f"Primary COTS candidate from retrieved data: {cots_analysis['recommended_product']}.")
        executive_parts.append(f"Suggested action: {modernization}")
        if products:
            executive_parts.append(
                f"Market intelligence: {len(products)} enterprise products retrieved via Tavily for {app.business_capability_l2}."
            )
        else:
            executive_parts.append("Market intelligence: no products retrieved in this run.")

        report_data = {
            "application": app.model_dump(),
            "time_analysis": time_analysis,
            "cots_analysis": cots_analysis,
            "swot_analysis": swot,
            "market_comparison": market_comparison,
            "modernization_recommendation": modernization,
            "rationalization_recommendation": time_analysis["decision"],
            "executive_report": " ".join(executive_parts),
        }
        return AgentResult(data=report_data)
