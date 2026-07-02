import json
import os
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import Session
from tavily import TavilyClient

from app.database.models import MarketProduct
from app.schemas import ApplicationInput


@dataclass
class StructuredMarketProduct:
    product_name: str
    vendor: str
    deployment_model: str
    key_capabilities: List[str]
    ai_features: List[str]
    target_enterprise_size: str
    licensing_model: str
    advantages: List[str]
    limitations: List[str]
    source_url: str
    source_title: str
    raw_snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


class MarketService:
    LISTICLE_PATTERNS = (
        r"^\d+\s+best",
        r"^top\s+\d+",
        r"best\s+\d+",
        r"list\s+for",
        r"comparison\s+guide",
        r"software\s+list",
    )

    KNOWN_PRODUCT_PATTERN = re.compile(
        r"\b("
        r"Salesforce(?:\s+Sales\s+Cloud)?|HubSpot(?:\s+CRM)?|Microsoft\s+Dynamics\s*365|"
        r"Zoho\s+CRM|Oracle\s+(?:CX|Sales|Cloud\s+CX)|SAP\s+(?:CRM|Sales\s+Cloud)|"
        r"ServiceNow|Pipedrive|Freshsales|Zendesk\s+Sell|Monday(?:\.com)?|"
        r"Workday|NetSuite|Infor|Epicor|Sage\s+Intacct|SugarCRM|Creatio"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, db: Session):
        self.db = db
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.client = TavilyClient(api_key=api_key) if api_key else None

    @staticmethod
    def _domain_key(app: ApplicationInput) -> str:
        parts = [
            app.business_capability_l1.strip().lower(),
            app.business_capability_l2.strip().lower(),
            app.business_capability_l3.strip().lower(),
            app.department.strip().lower(),
        ]
        return "|".join(parts)

    @staticmethod
    def _analyze_domain(app: ApplicationInput) -> dict:
        return {
            "l1": app.business_capability_l1.strip(),
            "l2": app.business_capability_l2.strip(),
            "l3": app.business_capability_l3.strip(),
            "department": app.department.strip(),
            "description": app.application_description.strip(),
            "technology_stack": app.technology_stack.strip(),
        }

    def _build_search_queries(self, app: ApplicationInput, domain: dict) -> List[str]:
        l1, l2, l3 = domain["l1"], domain["l2"], domain["l3"]
        dept = domain["department"]
        stack = domain["technology_stack"]

        return [
            f"{l2} enterprise COTS products Salesforce SAP Oracle Microsoft vendor comparison 2026",
            f"G2 {l2} software category leaders enterprise {dept}",
            f"enterprise {l2} SaaS platforms {l3} licensing deployment AI features",
            f"{l1} {l2} replacement for {stack} enterprise software vendors",
            f"best {l2} software for enterprise mid-market cloud on-prem 2026",
        ]

    @staticmethod
    def _is_listicle(title: str) -> bool:
        lower = title.lower().strip()
        return any(re.search(pattern, lower) for pattern in MarketService.LISTICLE_PATTERNS)

    @staticmethod
    def _normalize_product_name(name: str) -> str:
        cleaned = re.sub(r"\s+", " ", name.strip())
        return cleaned[:80]

    def _extract_named_products(self, content: str) -> List[str]:
        found = []
        seen = set()
        for match in self.KNOWN_PRODUCT_PATTERN.finditer(content):
            name = self._normalize_product_name(match.group(1))
            key = name.lower()
            if key not in seen:
                seen.add(key)
                found.append(name)
        return found

    @staticmethod
    def _vendor_from_product(product_name: str) -> str:
        vendor_map = {
            "salesforce": "Salesforce",
            "hubspot": "HubSpot",
            "microsoft": "Microsoft",
            "dynamics": "Microsoft",
            "zoho": "Zoho",
            "oracle": "Oracle",
            "sap": "SAP",
            "servicenow": "ServiceNow",
            "pipedrive": "Pipedrive",
            "freshsales": "Freshworks",
            "zendesk": "Zendesk",
            "monday": "monday.com",
            "workday": "Workday",
            "netsuite": "Oracle NetSuite",
            "infor": "Infor",
            "epicor": "Epicor",
            "sage": "Sage",
            "sugarcrm": "SugarCRM",
            "creatio": "Creatio",
        }
        lower = product_name.lower()
        for key, vendor in vendor_map.items():
            if key in lower:
                return vendor
        return product_name.split()[0]

    def _build_product_from_source(
        self,
        product_name: str,
        title: str,
        url: str,
        content: str,
    ) -> StructuredMarketProduct:
        vendor = self._vendor_from_product(product_name)
        capabilities = self._extract_capabilities(content)
        ai_features = self._extract_ai_features(content)
        advantages = self._extract_advantages(content)
        limitations = self._extract_limitations(content)

        if not capabilities:
            capabilities = [content[:180] + ("..." if len(content) > 180 else "")] if content else []

        return StructuredMarketProduct(
            product_name=product_name,
            vendor=vendor,
            deployment_model=self._detect_deployment(content),
            key_capabilities=capabilities,
            ai_features=ai_features or ["Not mentioned in retrieved source"],
            target_enterprise_size=self._detect_enterprise_size(content),
            licensing_model=self._detect_licensing(content),
            advantages=advantages or [f"Identified as a leading {product_name} option in retrieved enterprise sources"],
            limitations=limitations or ["Detailed limitations not available in retrieved source"],
            source_url=url,
            source_title=title,
            raw_snippet=content[:500],
        )

    @staticmethod
    def _extract_product_name(title: str) -> str:
        cleaned = title.strip()
        for delimiter in (" - ", " | ", ":", " – ", " — "):
            if delimiter in cleaned:
                return cleaned.split(delimiter)[0].strip()
        return cleaned[:100]

    @staticmethod
    def _extract_vendor(title: str, content: str) -> str:
        vendor_patterns = [
            r"by\s+([A-Z][A-Za-z0-9&.\- ]{2,40})",
            r"from\s+([A-Z][A-Za-z0-9&.\- ]{2,40})",
            r"([A-Z][A-Za-z0-9&.\- ]{2,40})\s+(?:CRM|ERP|platform|software|suite)",
        ]
        for pattern in vendor_patterns:
            match = re.search(pattern, title)
            if match:
                return match.group(1).strip()
        return MarketService._extract_product_name(title)

    @staticmethod
    def _detect_deployment(content: str) -> str:
        text = content.lower()
        has_cloud = any(k in text for k in ("cloud", "saas", "multi-tenant"))
        has_onprem = any(k in text for k in ("on-prem", "on prem", "on-premises", "self-hosted"))
        if has_cloud and has_onprem:
            return "Hybrid (Cloud + On-prem)"
        if has_cloud:
            return "Cloud/SaaS"
        if has_onprem:
            return "On-premises"
        return "Not specified in source"

    @staticmethod
    def _detect_licensing(content: str) -> str:
        text = content.lower()
        if "per user" in text or "per seat" in text:
            return "Per-user subscription"
        if "subscription" in text:
            return "Subscription"
        if "perpetual" in text or "one-time" in text:
            return "Perpetual license"
        if "consumption" in text or "usage-based" in text:
            return "Usage-based"
        return "Not specified in source"

    @staticmethod
    def _detect_enterprise_size(content: str) -> str:
        text = content.lower()
        if "enterprise" in text and "smb" in text:
            return "SMB to Enterprise"
        if "enterprise" in text or "large organization" in text:
            return "Enterprise"
        if "mid-market" in text or "mid market" in text:
            return "Mid-market"
        if "smb" in text or "small business" in text:
            return "SMB"
        return "Not specified in source"

    @staticmethod
    def _extract_list(content: str, keywords: List[str], limit: int = 4) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", content)
        hits = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in keywords) and 20 < len(sentence) < 220:
                hits.append(sentence.strip())
            if len(hits) >= limit:
                break
        return hits

    @staticmethod
    def _extract_capabilities(content: str) -> List[str]:
        capability_keywords = [
            "integration", "workflow", "automation", "analytics", "reporting",
            "security", "compliance", "api", "dashboard", "collaboration",
        ]
        return MarketService._extract_list(content, capability_keywords, limit=5)

    @staticmethod
    def _extract_ai_features(content: str) -> List[str]:
        ai_keywords = [
            "ai", "artificial intelligence", "machine learning", "copilot",
            "generative", "predictive", "intelligent", "llm",
        ]
        return MarketService._extract_list(content, ai_keywords, limit=4)

    @staticmethod
    def _extract_advantages(content: str) -> List[str]:
        advantage_keywords = [
            "leading", "scalable", "robust", "comprehensive", "trusted",
            "award", "leader", "best-in-class", "enterprise-grade",
        ]
        return MarketService._extract_list(content, advantage_keywords, limit=3)

    @staticmethod
    def _extract_limitations(content: str) -> List[str]:
        limitation_keywords = [
            "however", "limitation", "challenge", "complex", "costly",
            "steep learning", "customization", "vendor lock",
        ]
        return MarketService._extract_list(content, limitation_keywords, limit=3)

    def _structure_result(self, result: dict) -> List[StructuredMarketProduct]:
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        content = (result.get("content") or result.get("raw_content") or "").strip()
        if not title or not url or not content:
            return []

        named_products = self._extract_named_products(content)
        if named_products:
            return [
                self._build_product_from_source(name, title, url, content)
                for name in named_products
            ]

        if self._is_listicle(title):
            return []

        product_name = self._extract_product_name(title)
        return [self._build_product_from_source(product_name, title, url, content)]

    def _search_tavily(self, query: str, max_results: int = 5) -> List[dict]:
        if not self.client:
            return []
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=False,
            )
            return response.get("results", [])
        except Exception:
            return []

    def _fetch_from_tavily(self, app: ApplicationInput, limit: int = 10) -> List[StructuredMarketProduct]:
        domain = self._analyze_domain(app)
        queries = self._build_search_queries(app, domain)

        products: List[StructuredMarketProduct] = []
        seen_names = set()

        for query in queries:
            if len(products) >= limit:
                break
            results = self._search_tavily(query, max_results=5)
            for result in results:
                structured_items = self._structure_result(result)
                for structured in structured_items:
                    key = structured.product_name.lower()
                    if key in seen_names:
                        continue
                    seen_names.add(key)
                    products.append(structured)
                    if len(products) >= limit:
                        break

        return products

    def _load_cached(self, domain_key: str, limit: int) -> List[StructuredMarketProduct]:
        rows = (
            self.db.query(MarketProduct)
            .filter(MarketProduct.domain_key == domain_key)
            .order_by(MarketProduct.id.desc())
            .limit(limit)
            .all()
        )
        products = []
        for row in rows:
            if row.structured_json:
                try:
                    products.append(StructuredMarketProduct(**json.loads(row.structured_json)))
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass
            products.append(
                StructuredMarketProduct(
                    product_name=row.product_name,
                    vendor=row.vendor or row.product_name,
                    deployment_model="Not specified in source",
                    key_capabilities=[row.snippet or ""],
                    ai_features=["Not mentioned in retrieved source"],
                    target_enterprise_size="Not specified in source",
                    licensing_model="Not specified in source",
                    advantages=["Retrieved from cached market data"],
                    limitations=["Detailed limitations not available in cached source"],
                    source_url=row.source_url or "",
                    source_title=row.source_title or row.product_name,
                    raw_snippet=row.snippet or "",
                )
            )
        return products

    def _save_products(self, domain_key: str, query: str, products: List[StructuredMarketProduct]) -> None:
        for product in products:
            self.db.add(
                MarketProduct(
                    domain_key=domain_key,
                    query=query,
                    product_name=product.product_name,
                    vendor=product.vendor,
                    source_title=product.source_title,
                    source_url=product.source_url,
                    snippet=product.raw_snippet,
                    structured_json=json.dumps(product.to_dict()),
                )
            )
        self.db.commit()

    def get_products_for_application(self, app: ApplicationInput, limit: int = 10) -> List[StructuredMarketProduct]:
        domain_key = self._domain_key(app)
        cached = self._load_cached(domain_key, limit)
        if cached:
            return cached[:limit]

        products = self._fetch_from_tavily(app, limit=limit)
        if products:
            primary_query = self._build_search_queries(app, self._analyze_domain(app))[0]
            self._save_products(domain_key, primary_query, products)
        return products