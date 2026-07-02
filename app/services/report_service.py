from pathlib import Path
from typing import Iterable, List

import pandas as pd

from app.services.analysis_service import AnalysisResult


class ReportService:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_application_report(self, result: AnalysisResult) -> Path:
        safe_id = result.application_id.replace("/", "_").replace("\\", "_")
        report_path = self.output_dir / f"{safe_id}_executive_report.md"

        content = f"""# Executive APR Report: {result.application_name}

## Executive Summary
{result.executive_summary}

## Time Analysis
- TIM-E Score: **{result.tim_e_score}**
- Decision: **{result.tim_e_decision}**

## COTS Analysis
- COTS Score: **{result.cots_score}**
- Recommendation: **{result.cots_recommendation}**

## SWOT Analysis
- Strengths: {result.swot_strengths}
- Weaknesses: {result.swot_weaknesses}
- Opportunities: {result.swot_opportunities}
- Threats: {result.swot_threats}

## Market Comparison
{result.market_comparison}

## Recommendations
- Modernization Recommendation: {result.modernization_recommendation}
- Rationalization Recommendation: {result.rationalization_recommendation}
"""

        report_path.write_text(content, encoding="utf-8")
        return report_path

    def write_portfolio_summary(self, results: Iterable[AnalysisResult]) -> Path:
        rows: List[dict] = [item.to_dict() for item in results]
        df = pd.DataFrame(rows)

        csv_path = self.output_dir / "portfolio_summary.csv"
        df.to_csv(csv_path, index=False)

        markdown_path = self.output_dir / "portfolio_summary.md"
        decision_counts = df["tim_e_decision"].value_counts().to_dict()
        lines = [
            "# APR Portfolio Summary",
            "",
            f"- Total applications analyzed: **{len(df)}**",
            f"- Invest: **{decision_counts.get('Invest', 0)}**",
            f"- Migrate: **{decision_counts.get('Migrate', 0)}**",
            f"- Tolerate: **{decision_counts.get('Tolerate', 0)}**",
            f"- Eliminate: **{decision_counts.get('Eliminate', 0)}**",
            "",
            "## Top 10 by TIM-E Score",
            "",
        ]

        top10 = df.sort_values("tim_e_score", ascending=False).head(10)
        for _, row in top10.iterrows():
            lines.append(
                f"- {row['application_name']} ({row['application_id']}): "
                f"{row['tim_e_score']} | {row['tim_e_decision']}"
            )

        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return markdown_path
