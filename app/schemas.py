from pydantic import BaseModel


class ApplicationInput(BaseModel):
    application_id: str
    application_name: str
    owner: str
    owner_email: str
    department: str
    application_description: str
    application_status: str
    business_criticality: str
    business_fitness: str
    strategic_relevance: str
    usage_adoption: str
    functional_redundancy: str
    application_security_level: str
    maintainability: str
    application_stability: str
    skill_availability: str
    availability: str
    reliability: str
    scalability: str
    technology_stack: str
    annual_fte_cost: float
    annual_license_cost: float
    fte_count: int
    annual_infrastructure_cost: float
    other_costs: float
    business_capability_l1: str
    business_capability_l2: str
    business_capability_l3: str
