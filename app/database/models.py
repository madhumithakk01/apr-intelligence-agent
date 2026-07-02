from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.sql import func

from app.database.db import Base


class Application(Base):

    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)

    application_id = Column(String, unique=True)

    application_name = Column(String)

    owner = Column(String)

    owner_email = Column(String)

    department = Column(String)

    application_description = Column(String)

    application_status = Column(String)

    business_criticality = Column(String)

    business_fitness = Column(String)

    strategic_relevance = Column(String)

    usage_adoption = Column(String)

    functional_redundancy = Column(String)

    application_security_level = Column(String)

    maintainability = Column(String)

    application_stability = Column(String)

    skill_availability = Column(String)

    availability = Column(String)

    reliability = Column(String)

    scalability = Column(String)

    technology_stack = Column(String)

    annual_fte_cost = Column(Float)

    annual_license_cost = Column(Float)

    fte_count = Column(Integer)

    annual_infrastructure_cost = Column(Float)

    other_costs = Column(Float)

    business_capability_l1 = Column(String)

    business_capability_l2 = Column(String)

    business_capability_l3 = Column(String)


class MarketProduct(Base):

    __tablename__ = "market_products"

    id = Column(Integer, primary_key=True, index=True)
    domain_key = Column(String, index=True)
    query = Column(String, index=True)
    product_name = Column(String, index=True)
    vendor = Column(String)
    source_title = Column(String)
    source_url = Column(String)
    snippet = Column(Text)
    structured_json = Column(Text)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AnalysisRun(Base):

    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String, index=True)
    application_name = Column(String)
    tim_e_decision = Column(String)
    tim_e_score = Column(Float)
    cots_recommendation = Column(String)
    modernization_recommendation = Column(String)
    report_markdown = Column(Text)
    report_pdf_path = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)