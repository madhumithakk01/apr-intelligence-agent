from pathlib import Path

from app.services.analysis_service import AnalysisService
from app.ingestion.excel_loader import ExcelLoader
from app.services.report_service import ReportService


def run(dataset_path: str = "data/Dataset.xlsx", output_dir: str = "reports") -> None:
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    loader = ExcelLoader(str(dataset))
    df = loader.load()
    if df.empty:
        raise ValueError("No rows found in dataset.")

    analysis_service = AnalysisService()
    report_service = ReportService(output_dir=output_dir)

    results = []
    for _, row in df.iterrows():
        result = analysis_service.analyze(row)
        report_service.write_application_report(result)
        results.append(result)

    summary_path = report_service.write_portfolio_summary(results)
    print(f"Analyzed {len(results)} applications.")
    print(f"Reports generated in: {Path(output_dir).resolve()}")
    print(f"Portfolio summary: {summary_path.resolve()}")


if __name__ == "__main__":
    run()
