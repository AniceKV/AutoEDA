"""
pdf_report_generator.py
-----------------------
Generates a professional, publication-grade PDF report (eda_report.pdf)
synthesizing canonical metrics.json and executive summary telemetry.
"""

import json
import os
import re
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def generate_pdf_report(workspace_dir: str, output_pdf_path: Optional[str] = None) -> str:
    """
    Generates a publication-quality PDF report for the given workspace directory.

    Args:
        workspace_dir: Directory containing metrics.json and summary_report.md.
        output_pdf_path: Optional custom path to save the PDF. Defaults to eda_report.pdf in workspace_dir.

    Returns:
        Absolute path to the created PDF file.
    """
    if not output_pdf_path:
        output_pdf_path = os.path.join(workspace_dir, "eda_report.pdf")

    if not HAS_REPORTLAB:
        print("[pdf_report_generator] Warning: reportlab is not installed. PDF generation skipped.")
        return output_pdf_path

    metrics_path = os.path.join(workspace_dir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    summary_path = os.path.join(workspace_dir, "summary_report.md")
    summary_text = ""
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()
        except Exception:
            summary_text = ""

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#4f46e5")  # Indigo
    dark_text = colors.HexColor("#0f172a")       # Slate 900
    dim_text = colors.HexColor("#475569")        # Slate 600
    bg_light = colors.HexColor("#f8fafc")        # Slate 50

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=dim_text,
        spaceAfter=14
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=dark_text,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=dark_text
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=dark_text
    )

    elements = []

    # Title & Subtitle Header
    dataset_name = metrics.get("dataset_info", {}).get("dataset_name") or os.path.basename(workspace_dir)
    elements.append(Paragraph("AutoEDA Pro — Statistical & Profiling Report", title_style))
    elements.append(Paragraph(f"Dataset: <b>{dataset_name}</b> &bull; Autonomous Machine Learning Telemetry", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=14))

    # 1. Dataset Executive Overview
    elements.append(Paragraph("1. Dataset Overview", h2_style))
    num_rows = metrics.get("dataset_info", {}).get("num_rows", 0)
    num_cols = metrics.get("dataset_info", {}).get("num_cols", 0)
    missing_pct = metrics.get("missing_summary", {}).get("total_missing_percentage", 0.0)

    overview_data = [
        [Paragraph("Total Rows", table_cell_bold), Paragraph(f"{num_rows:,}", table_cell_style),
         Paragraph("Total Columns", table_cell_bold), Paragraph(str(num_cols), table_cell_style)],
        [Paragraph("Missing Cells (%)", table_cell_bold), Paragraph(f"{missing_pct:.2f}%", table_cell_style),
         Paragraph("Target Column", table_cell_bold), Paragraph(str(metrics.get("target_column") or "N/A"), table_cell_style)]
    ]

    t_overview = Table(overview_data, colWidths=[1.5 * inch, 2.0 * inch, 1.5 * inch, 2.0 * inch])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(t_overview)
    elements.append(Spacer(1, 14))

    # 2. Executive Summary Text (if present)
    if summary_text.strip():
        elements.append(Paragraph("2. Executive Summary Narrative", h2_style))
        clean_summary = re.sub(r'#+\s*', '', summary_text)
        for line in clean_summary.split("\n"):
            line = line.strip()
            if line:
                elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 14))

    # 3. Variable Profiles Table
    col_profiles = metrics.get("column_profiles", [])
    if col_profiles:
        elements.append(Paragraph("3. Variable Deep-Dive Summary", h2_style))
        headers = ["Column", "Type", "Missing", "Unique", "Mean / Top", "Std Dev"]
        data = [[Paragraph(h, table_header_style) for h in headers]]

        for c in col_profiles[:25]:  # Top 25 variables
            col_name = str(c.get("column", ""))
            dtype = str(c.get("dtype", ""))
            missing = f"{c.get('missing_count', 0)} ({c.get('missing_pct', 0)}%)"
            unique = str(c.get("cardinality", ""))
            mean_val = f"{c.get('mean', 'N/A'):.2f}" if isinstance(c.get('mean'), (int, float)) else "N/A"
            std_val = f"{c.get('std', 'N/A'):.2f}" if isinstance(c.get('std'), (int, float)) else "N/A"

            data.append([
                Paragraph(col_name, table_cell_bold),
                Paragraph(dtype, table_cell_style),
                Paragraph(missing, table_cell_style),
                Paragraph(unique, table_cell_style),
                Paragraph(mean_val, table_cell_style),
                Paragraph(std_val, table_cell_style)
            ])

        t_vars = Table(data, colWidths=[1.8 * inch, 0.9 * inch, 1.2 * inch, 0.8 * inch, 1.1 * inch, 1.2 * inch])
        t_vars.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(t_vars)
        elements.append(Spacer(1, 14))

    # 4. Statistical Hypothesis Testing
    hyp_tests = metrics.get("statistical_hypothesis_tests", {})
    ranked_details = hyp_tests.get("ranked_significant_details", [])
    if ranked_details:
        elements.append(Paragraph("4. Statistical Hypothesis Testing Results", h2_style))
        h_headers = ["Feature", "Test Conducted", "Statistic", "P-Value", "Significant"]
        h_data = [[Paragraph(h, table_header_style) for h in h_headers]]

        for r in ranked_details[:15]:
            feat = str(r.get("feature", "N/A"))
            test_name = str(r.get("test") or r.get("test_name") or "Statistical Test")
            stat = r.get("statistic") or r.get("effect_size") or "N/A"
            stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else str(stat)
            pval = r.get("p_value", "N/A")
            pval_str = f"{pval:.4e}" if isinstance(pval, (int, float)) and pval < 0.0001 else (f"{pval:.4f}" if isinstance(pval, (int, float)) else str(pval))
            is_sig = "YES (α=0.05)" if r.get("is_statistically_significant", True) else "No"

            h_data.append([
                Paragraph(feat, table_cell_bold),
                Paragraph(test_name, table_cell_style),
                Paragraph(stat_str, table_cell_style),
                Paragraph(pval_str, table_cell_style),
                Paragraph(is_sig, table_cell_bold if "YES" in is_sig else table_cell_style)
            ])

        t_hyp = Table(h_data, colWidths=[1.6 * inch, 2.0 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch])
        t_hyp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#312e81")),  # Dark Indigo
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(t_hyp)
        elements.append(Spacer(1, 14))

    # 5. Predictive Modeling Blueprint
    blueprint = metrics.get("predictive_modeling_blueprint", {})
    if blueprint:
        elements.append(Paragraph("5. Predictive Modeling Strategy Blueprint", h2_style))
        bp_exec = blueprint.get("executive_summary", "")
        if bp_exec:
            elements.append(Paragraph(bp_exec, body_style))

        task_type = str(blueprint.get("task_type", "N/A"))
        recommended_models = ", ".join(blueprint.get("recommended_models", [])) or "N/A"
        cv_strategy = str(blueprint.get("cross_validation_strategy", "5-Fold Stratified K-Fold"))

        bp_data = [
            [Paragraph("Task Type", table_cell_bold), Paragraph(task_type.upper(), table_cell_style)],
            [Paragraph("Recommended Models", table_cell_bold), Paragraph(recommended_models, table_cell_style)],
            [Paragraph("Validation Strategy", table_cell_bold), Paragraph(cv_strategy, table_cell_style)],
        ]
        t_bp = Table(bp_data, colWidths=[2.2 * inch, 4.8 * inch])
        t_bp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(t_bp)

    doc.build(elements)
    print(f"[pdf_report_generator] PDF report generated successfully: {output_pdf_path}")
    return output_pdf_path
