from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from app.api.batch import router as batch_router
from app.api.shadow import router as shadow_router
from app.database.db import Base
from app.database.db import SessionLocal
from app.database.db import engine
from app.database.db import migrate_schema
from app.database.models import AnalysisRun
from app.schemas import ApplicationInput
from app.services.agent_service import APRAgentService
from app.services.report_renderer import ReportRenderer

import app.database.models  # noqa: F401

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(bind=engine)
migrate_schema()

app = FastAPI(
    title="APR Intelligence Agent",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
renderer = ReportRenderer(reports_dir=str(REPORTS_DIR))

# The portfolio-scale pipeline (CLAUDE.md section 5) runs as an async job
# -- submit, poll, resume -- because a 100-row run is rate-limited well
# past any request timeout (section 11). The single-record routes below
# are the legacy path, kept until callers move over.
app.include_router(batch_router)
app.include_router(shadow_router)


def _run_analysis(db, app_input: ApplicationInput) -> tuple[dict, str, str, AnalysisRun]:
    report_data = APRAgentService(db).analyze(app_input).data
    report_markdown = renderer.markdown(report_data)
    safe_id = app_input.application_id.replace("/", "_").replace("\\", "_")
    pdf_path = renderer.write_pdf(application_id=safe_id, report=report_data)

    run = AnalysisRun(
        application_id=app_input.application_id,
        application_name=app_input.application_name,
        tim_e_decision=report_data["time_analysis"]["decision"],
        tim_e_score=report_data["time_analysis"]["score"],
        cots_recommendation=report_data["cots_analysis"]["recommendation"],
        modernization_recommendation=report_data["modernization_recommendation"],
        report_markdown=report_markdown,
        report_pdf_path=pdf_path,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return report_data, report_markdown, pdf_path, run


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze")
def analyze_application(
    request: Request,
    application_id: str = Form(...),
    application_name: str = Form(...),
    owner: str = Form(...),
    owner_email: str = Form(...),
    department: str = Form(...),
    application_description: str = Form(...),
    application_status: str = Form(...),
    business_criticality: str = Form(...),
    business_fitness: str = Form(...),
    strategic_relevance: str = Form(...),
    usage_adoption: str = Form(...),
    functional_redundancy: str = Form(...),
    application_security_level: str = Form(...),
    maintainability: str = Form(...),
    application_stability: str = Form(...),
    skill_availability: str = Form(...),
    availability: str = Form(...),
    reliability: str = Form(...),
    scalability: str = Form(...),
    technology_stack: str = Form(...),
    annual_fte_cost: Optional[float] = Form(default=None),
    annual_license_cost: Optional[float] = Form(default=None),
    fte_count: Optional[int] = Form(default=None),
    annual_infrastructure_cost: Optional[float] = Form(default=None),
    other_costs: Optional[float] = Form(default=None),
    business_capability_l1: str = Form(...),
    business_capability_l2: str = Form(...),
    business_capability_l3: str = Form(...),
):
    app_input = ApplicationInput(
        application_id=application_id.strip(),
        application_name=application_name.strip(),
        owner=owner.strip(),
        owner_email=owner_email.strip(),
        department=department.strip(),
        application_description=application_description.strip(),
        application_status=application_status.strip(),
        business_criticality=business_criticality.strip(),
        business_fitness=business_fitness.strip(),
        strategic_relevance=strategic_relevance.strip(),
        usage_adoption=usage_adoption.strip(),
        functional_redundancy=functional_redundancy.strip(),
        application_security_level=application_security_level.strip(),
        maintainability=maintainability.strip(),
        application_stability=application_stability.strip(),
        skill_availability=skill_availability.strip(),
        availability=availability.strip(),
        reliability=reliability.strip(),
        scalability=scalability.strip(),
        technology_stack=technology_stack.strip(),
        annual_fte_cost=annual_fte_cost,
        annual_license_cost=annual_license_cost,
        fte_count=fte_count,
        annual_infrastructure_cost=annual_infrastructure_cost,
        other_costs=other_costs,
        business_capability_l1=business_capability_l1.strip(),
        business_capability_l2=business_capability_l2.strip(),
        business_capability_l3=business_capability_l3.strip(),
    )

    db = SessionLocal()
    try:
        report_data, report_markdown, _, run = _run_analysis(db, app_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        db.close()

    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "report": report_data,
            "report_markdown": report_markdown,
            "run_id": run.id,
        },
    )


@app.post("/api/analyze")
def analyze_application_api(payload: ApplicationInput):
    db = SessionLocal()
    try:
        report_data, report_markdown, pdf_path, run = _run_analysis(db, payload)
        return {
            "run_id": run.id,
            "report": report_data,
            "pdf_download": f"/download/{run.id}",
            "pdf_path": pdf_path,
            "report_markdown": report_markdown,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        db.close()


@app.get("/download/{run_id}")
def download_report(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Report not found")
        if not run.report_pdf_path or not Path(run.report_pdf_path).exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        return FileResponse(
            path=run.report_pdf_path,
            filename=f"{run.application_id}_apr_report.pdf",
            media_type="application/pdf",
        )
    finally:
        db.close()
