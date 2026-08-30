from dataclasses import dataclass, field
from typing import List

from app.database.db import Base, engine, migrate_schema
from app.database.models import Application
from app.ingestion.cost_parsing import build_numeric_field_notes, parse_cost_cell, parse_fte_count
from app.ingestion.excel_loader import ExcelLoader
from app.services.import_service import ImportService

COST_FIELDS = {
    "annual_fte_cost": "Annual FTE Cost",
    "annual_license_cost": "Annual License Cost",
    "annual_infrastructure_cost": "Annual Infrastructure Cost",
    "other_costs": "Other Costs",
}


@dataclass
class ImportSummary:
    imported: int = 0
    collisions: List[dict] = field(default_factory=list)


def run(dataset_path: str = "data/Dataset.xlsx") -> ImportSummary:
    Base.metadata.create_all(bind=engine)
    migrate_schema()

    loader = ExcelLoader(dataset_path)
    df = loader.load()

    intra_file_collisions = ExcelLoader.find_duplicate_application_ids(df)
    skip_ids = set(intra_file_collisions.keys())

    summary = ImportSummary()
    for application_id, occurrences in intra_file_collisions.items():
        summary.collisions.append(
            {"application_id": application_id, "source": "duplicate-in-file", "occurrences": occurrences}
        )

    service = ImportService()

    for _, row in df.iterrows():
        application_id = str(row["Application ID"]).strip()
        if application_id in skip_ids:
            continue

        if service.application_exists(application_id):
            summary.collisions.append(
                {
                    "application_id": application_id,
                    "source": "existing-in-db",
                    "occurrences": [
                        {
                            "source_sheet": row.get("source_sheet"),
                            "source_row": row.get("source_row"),
                            "application_name": row.get("Application Name"),
                        }
                    ],
                }
            )
            continue

        parsed = {
            field_name: parse_cost_cell(row.get(column), field_name=field_name, application_id=application_id)
            for field_name, column in COST_FIELDS.items()
        }
        parsed["fte_count"] = parse_fte_count(row.get("FTE Count"), application_id=application_id)

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
            annual_fte_cost=parsed["annual_fte_cost"].value,
            annual_license_cost=parsed["annual_license_cost"].value,
            fte_count=int(round(parsed["fte_count"].value)) if parsed["fte_count"].value is not None else None,
            annual_infrastructure_cost=parsed["annual_infrastructure_cost"].value,
            other_costs=parsed["other_costs"].value,
            business_capability_l1=row["Business Capability L1"],
            business_capability_l2=row["Business Capability L2"],
            business_capability_l3=row["Business Capability L3"],
            numeric_field_notes=build_numeric_field_notes(parsed),
        )

        service.save(application)
        summary.imported += 1

    service.commit()
    service.close()
    return summary


if __name__ == "__main__":
    result = run()
    print(f"{result.imported} applications imported.")
    if result.collisions:
        print(f"WARNING: {len(result.collisions)} Application ID collision(s) -- not imported, needs manual resolution:")
        for collision in result.collisions:
            print(f"  - {collision}")
