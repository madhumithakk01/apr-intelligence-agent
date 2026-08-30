from dataclasses import dataclass, field
from typing import List

from app.database.db import Base, engine, migrate_schema
from app.database.models import Application
from app.ingestion.excel_loader import ExcelLoader
from app.ingestion.row_mapping import build_application_fields
from app.services.import_service import ImportService


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

        fields = build_application_fields(row, application_id=application_id)
        application = Application(**fields)

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
