from app.database.models import Application
from app.services.excel_loader import ExcelLoader
from app.services.import_service import ImportService


loader = ExcelLoader("data/Dataset.xlsx")

df = loader.load()

service = ImportService()

count = 0

for _, row in df.iterrows():

    application_id = str(row["Application ID"]).strip()

    if service.application_exists(application_id):
        continue

    application = Application(

        application_id=application_id,

        application_name=row["Application Name"],

        owner=row["Owner"],

        owner_email=row["Owner Email"],

        department=row["Department"],

        application_description=row["Application Description"],

        application_status=row["Application Status"],

        business_criticality=row["Business Criticality"],

        business_fitness=row["Business Fitness"],

        strategic_relevance=row["Strategic Relevance"],

        usage_adoption=row["Usage & Adoption"],

        functional_redundancy=row["Functional redundancy"],

        application_security_level=row["Application Security Level"],

        maintainability=row["Maintainability"],

        application_stability=row["Application Stability"],

        skill_availability=row["Skill availability"],

        availability=row["Availability"],

        reliability=row["Reliability"],

        scalability=row["Scalability"],

        technology_stack=row["Technology Stack"],

        annual_fte_cost=float(row.get("Annual FTE Cost", 0) or 0),

        annual_license_cost=float(row.get("Annual License Cost", 0) or 0),

        fte_count=int(row.get("FTE Count", 0) or 0),

        annual_infrastructure_cost=float(row.get("Annual Infrastructure Cost", 0) or 0),

        other_costs=float(row.get("Other Costs", 0) or 0),

        business_capability_l1=row["Business Capability L1"],

        business_capability_l2=row["Business Capability L2"],

        business_capability_l3=row["Business Capability L3"]

    )

    service.save(application)

    count += 1

service.commit()

service.close()

print(f"{count} applications imported.")