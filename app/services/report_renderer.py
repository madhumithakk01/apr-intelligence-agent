import html
from pathlib import Path
from typing import Dict
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer


class ReportRenderer:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _esc(text: str) -> str:
        return html.escape(str(text or ""))

    @staticmethod
    def _bullet_lines(items: List[str]) -> str:
        if not items:
            return "Not available"
        return "<br/>".join(f"• {html.escape(item)}" for item in items)

    def markdown(self, report: Dict) -> str:
        market_lines = []
        for item in report["market_comparison"]:
            market_lines.append(
                f"### {item['product_name']} ({item['vendor']})\n"
                f"- Deployment: {item['deployment_model']}\n"
                f"- Licensing: {item['licensing_model']}\n"
                f"- Target: {item['target_enterprise_size']}\n"
                f"- Capabilities: {', '.join(item['key_capabilities'][:3])}\n"
                f"- Source: {item['source_url']}\n"
            )

        return f"""# APR Executive Report

## Application
- Application ID: {report['application']['application_id']}
- Application Name: {report['application']['application_name']}
- Department: {report['application']['department']}

## Executive Summary
{report['executive_report']}

## Time Analysis
- TIM-E Score: **{report['time_analysis']['score']}**
- Decision: **{report['time_analysis']['decision']}**

## COTS Analysis
- COTS Score: **{report['cots_analysis']['score']}**
- Recommendation: **{report['cots_analysis']['recommendation']}**

## SWOT Analysis
- Strengths: {report['swot_analysis']['strengths']}
- Weaknesses: {report['swot_analysis']['weaknesses']}
- Opportunities: {report['swot_analysis']['opportunities']}
- Threats: {report['swot_analysis']['threats']}
- Basis: {report['swot_analysis']['comparison_basis']}

## Market Comparison (Retrieved Products)
{chr(10).join(market_lines) if market_lines else "- No market products retrieved in this run."}

## Recommendations
- Modernization: {report['modernization_recommendation']}
- Rationalization: {report['rationalization_recommendation']}
"""

    def write_pdf(self, application_id: str, report: Dict) -> str:
        pdf_path = self.reports_dir / f"{application_id}_report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor("#14213d"),
        )
        heading_style = ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#0f62fe"),
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )

        story = []
        story.append(Paragraph("APR Executive Report", title_style))
        story.append(Paragraph(
            f"<b>Application:</b> {self._esc(report['application']['application_name'])} "
            f"({self._esc(report['application']['application_id'])})<br/>"
            f"<b>Department:</b> {self._esc(report['application']['department'])}",
            body_style,
        ))
        story.append(Spacer(1, 8))

        sections = [
            ("Executive Summary", report["executive_report"]),
            (
                "Time Analysis",
                f"TIM-E Score: {report['time_analysis']['score']}<br/>"
                f"Decision: <b>{self._esc(report['time_analysis']['decision'])}</b>",
            ),
            (
                "COTS Analysis",
                f"COTS Score: {report['cots_analysis']['score']}<br/>"
                f"Recommendation: {self._esc(report['cots_analysis']['recommendation'])}"
                + (
                    f"<br/>Recommended Product: {self._esc(report['cots_analysis']['recommended_product'])}"
                    if report["cots_analysis"].get("recommended_product")
                    else ""
                ),
            ),
            (
                "SWOT Analysis",
                f"<b>Strengths:</b> {self._esc(report['swot_analysis']['strengths'])}<br/>"
                f"<b>Weaknesses:</b> {self._esc(report['swot_analysis']['weaknesses'])}<br/>"
                f"<b>Opportunities:</b> {self._esc(report['swot_analysis']['opportunities'])}<br/>"
                f"<b>Threats:</b> {self._esc(report['swot_analysis']['threats'])}<br/>"
                f"<b>Basis:</b> {self._esc(report['swot_analysis']['comparison_basis'])}",
            ),
            (
                "Recommendations",
                f"<b>Modernization:</b> {self._esc(report['modernization_recommendation'])}<br/>"
                f"<b>Rationalization:</b> {self._esc(report['rationalization_recommendation'])}",
            ),
        ]

        for heading, content in sections:
            story.append(Paragraph(heading, heading_style))
            story.append(Paragraph(content, body_style))

        story.append(Paragraph("Market Comparison (Retrieved Products)", heading_style))
        if report["market_comparison"]:
            for idx, product in enumerate(report["market_comparison"], start=1):
                product_block = (
                    f"<b>{idx}. {self._esc(product['product_name'])}</b> "
                    f"by {self._esc(product['vendor'])}<br/>"
                    f"Deployment: {self._esc(product['deployment_model'])} | "
                    f"Licensing: {self._esc(product['licensing_model'])} | "
                    f"Target: {self._esc(product['target_enterprise_size'])}<br/>"
                    f"<b>Capabilities:</b><br/>{self._bullet_lines(product['key_capabilities'])}<br/>"
                    f"<b>AI Features:</b><br/>{self._bullet_lines(product['ai_features'])}<br/>"
                    f"<b>Advantages:</b><br/>{self._bullet_lines(product['advantages'])}<br/>"
                    f"<b>Limitations:</b><br/>{self._bullet_lines(product['limitations'])}<br/>"
                    f"Source: {self._esc(product['source_url'])}"
                )
                story.append(Paragraph(product_block, body_style))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(
                "No market products were retrieved in this run. Re-run analysis after verifying Tavily API connectivity.",
                body_style,
            ))

        doc.build(story)
        return str(pdf_path)
