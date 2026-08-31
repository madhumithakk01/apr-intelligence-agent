from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text

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

    numeric_field_notes = Column(Text)
