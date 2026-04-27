from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = Path("reports/recomart_assignment_report.pdf")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"

PAGE_WIDTH, PAGE_HEIGHT = A4


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F4E79"),
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2F6B8F"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#222222"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#222222"),
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "table_cell": ParagraphStyle(
            "TableCellCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#222222"),
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.5,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "check": ParagraphStyle(
            "CheckCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#222222"),
        ),
    }

    return styles


def paragraph(text: str, styles: dict[str, ParagraphStyle], style_name: str = "body") -> Paragraph:
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe_text, styles[style_name])


def bullet_list(items: Iterable[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, styles, "body"), leftIndent=12) for item in items],
        bulletType="bullet",
        leftIndent=18,
    )


def df_to_table(
    df: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
    max_rows: int = 12,
    max_cols: int = 6,
    column_widths: list[float] | None = None,
) -> Table | Paragraph:
    if df.empty:
        return paragraph("Not available.", styles)

    display_df = df.head(max_rows).iloc[:, :max_cols].copy()

    data: list[list[Paragraph]] = []
    header_row = [Paragraph(str(col), styles["table_header"]) for col in display_df.columns]
    data.append(header_row)

    for _, row in display_df.iterrows():
        data.append([Paragraph(clean_value(value), styles["table_cell"]) for value in row])

    if column_widths is None:
        available_width = 7.1 * inch
        column_widths = [available_width / len(display_df.columns)] * len(display_df.columns)

    table = Table(data, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ]
        )
    )

    return table


def choose_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def validation_warning_details(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    return choose_columns(
        df[df["status"] == "WARN"].copy(),
        ["dataset_name", "check_name", "status", "details"],
    )


def image_block(
    image_path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    width: float = 6.4 * inch,
) -> list:
    if not image_path.exists():
        return [paragraph(f"Missing image: {image_path}", styles)]

    img = Image(str(image_path))
    scale = width / img.drawWidth
    img.drawWidth = width
    img.drawHeight = img.drawHeight * scale

    return [
        img,
        Paragraph(caption, styles["caption"]),
    ]


def section_title(title: str, story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(title, styles["h1"]))


def subsection_title(title: str, story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(title, styles["h2"]))


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(PAGE_WIDTH - 0.55 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    styles = make_styles()
    story: list = []

    retail_external = read_csv("reports/retailrocket_external_summary.csv")
    retail_events = read_csv("reports/retailrocket_event_type_summary.csv")
    raw_validation = read_csv("reports/data_quality/retailrocket_raw_validation_report.csv")
    retail_validation = read_csv("reports/data_quality/retailrocket_staged_validation_report.csv")
    preparation = read_csv("reports/retailrocket_preparation_summary.csv")
    curated = read_csv("reports/retailrocket_curated_summary.csv")
    retail_warehouse = read_csv("reports/retailrocket_duckdb_load_summary.csv")
    retail_features = read_csv("reports/retailrocket_feature_summary.csv")
    retail_training = read_csv("reports/retailrocket_model_training_summary.csv")
    retail_content_training = read_csv("reports/retailrocket_content_model_training_summary.csv")
    retail_model_comparison = read_csv("reports/retailrocket_model_comparison.csv")
    retail_inference = read_csv("reports/retailrocket_inference_demo_summary.csv")
    retail_orchestration = read_csv("reports/orchestration_retailrocket_pipeline_summary.csv")

    sql_schema = read_csv("reports/sql_schema_summary.csv")
    feature_logic = read_csv("reports/feature_logic_summary.csv")
    feature_metadata = read_csv("reports/feature_metadata_documentation.csv")
    dvc_summary = read_csv("reports/dvc_versioning_summary.csv")
    repo_structure = read_csv("reports/repository_structure_summary.csv")
    feature_retrieval = read_csv("reports/feature_store_retrieval_demo.csv")

    story.append(Spacer(1, 1.0 * inch))
    story.append(Paragraph("RecoMart Retailrocket Recommendation Pipeline", styles["title"]))
    story.append(
        Paragraph(
            "Data Management for Machine Learning - Assignment I<br/>"
            "End-to-End Data Management Pipeline for a Recommendation System",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        paragraph(
            "This PDF is the single consolidated documentation report. It covers problem formulation, "
            "data ingestion, raw storage, validation, preparation, EDA, SQL transformation, feature store, "
            "DVC versioning, MLflow tracking, model training, inference, orchestration, and reproducibility.",
            styles,
        )
    )
    story.append(PageBreak())

    section_title("1. Assignment Requirement Coverage", story, styles)
    coverage_rows = [
        ["Problem formulation", "Covered", "Business problem, objectives, outputs, metrics"],
        ["Data collection and ingestion", "Covered", "Retailrocket batch CSV data and REST/mock catalog metadata delta source"],
        ["Raw data storage", "Covered", "Partitioned data lake using source/type/ingestion_timestamp"],
        ["Data profiling and validation", "Covered", "Raw, staged, and curated validation scripts and reports"],
        ["Data preparation and EDA", "Covered", "Staged Parquet, curated datasets, and summary plots"],
        ["Feature engineering and transformation", "Covered", "DuckDB SQL warehouse and feature tables"],
        ["Feature store", "Covered", "Custom registry and retrieval demo"],
        ["Data versioning and lineage", "Covered", "DVC metadata and versioning workflow"],
        ["Model training and evaluation", "Covered", "Retailrocket popularity baseline and content-based recommender with HitRate, Precision, Recall, NDCG"],
        ["Pipeline orchestration", "Covered", "Prefect flows and successful run logs"],
        ["Documentation and demo", "Covered", "Single PDF report, screenshots, video script"],
    ]
    coverage_df = pd.DataFrame(coverage_rows, columns=["Requirement", "Status", "Evidence"])
    story.append(df_to_table(coverage_df, styles, max_rows=20, max_cols=3, column_widths=[2.2 * inch, 1.0 * inch, 3.8 * inch]))

    section_title("2. Problem Formulation", story, styles)
    story.append(
        paragraph(
            "RecoMart is an e-commerce startup that wants product recommendations to improve customer "
            "engagement, conversion, and cross-selling. The pipeline produces clean datasets, engineered "
            "features, trained recommendation models, inference outputs, and reproducible metadata for "
            "monitoring and lineage.",
            styles,
        )
    )
    story.append(
        bullet_list(
            [
                "Business problem: recommend relevant products to users based on interaction history and item metadata.",
                "Expected outputs: clean EDA-ready datasets, feature tables, trained model artifact, inference interface.",
                "Evaluation metrics: Precision@K, Recall@K, HitRate@K, and NDCG@K.",
                "Implemented models: Retailrocket event-weighted popularity baseline and category/content-based recommender.",
            ],
            styles,
        )
    )

    section_title("3. Data Sources", story, styles)
    subsection_title("3.1 Retailrocket External Dataset", story, styles)
    story.append(
        paragraph(
            "Retailrocket is the main large-scale dataset. It includes visitor-item events, item properties, "
            "and category hierarchy data.",
            styles,
        )
    )
    story.append(df_to_table(retail_external, styles, max_rows=3, max_cols=8))
    story.append(Spacer(1, 0.1 * inch))
    story.append(df_to_table(retail_events, styles, max_rows=5, max_cols=5))

    section_title("4. Ingestion, Raw Storage, and Logging", story, styles)
    story.append(
        paragraph(
            "The ingestion layer contains Retailrocket batch ingestion and Retailrocket REST/mock "
            "catalog metadata deltas. Retailrocket source data is tracked by DVC as an external dataset, and "
            "raw ingestion snapshots are stored in a structured local data lake layout. The teammate-hosted real "
            "API returns non-repeated rows using the data/success/count/total_rows/unread_rows contract, while "
            "mock mode remains the reproducible local fallback using records/next_cursor/has_more.",
            styles,
        )
    )
    story.append(
        bullet_list(
            [
                "Retailrocket inspection script: src/ingestion/inspect_retailrocket_external.py",
                "Retailrocket batch ingestion script: src/ingestion/ingest_retailrocket_batch.py",
                "Retailrocket API ingestion script: src/ingestion/ingest_retailrocket_catalog_api.py",
                "Mock API server: src/ingestion/mock_retailrocket_catalog_api.py",
                "Real API mode: set RECOMART_CATALOG_API_MOCK_MODE=false, RECOMART_CATALOG_API_BASE_URL, RECOMART_CATALOG_API_ENDPOINT, and RECOMART_CATALOG_API_PAGE_SIZE.",
                "API page size: RECOMART_CATALOG_API_PAGE_SIZE is sent as the count query parameter in real API mode.",
                "Raw layout: data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/",
                "Raw batch source: data/raw/source=retailrocket_batch/",
                "Raw API source: data/raw/source=retailrocket_api/",
                "External storage layout: data/external/retailrocket",
            ],
            styles,
        )
    )

    section_title("5. Data Validation and Quality Reports", story, styles)
    subsection_title("5.1 Raw Validation", story, styles)
    if not raw_validation.empty and "status" in raw_validation.columns:
        story.append(df_to_table(raw_validation.groupby("status").size().reset_index(name="check_count"), styles))
    else:
        story.append(paragraph("Raw validation summary not available.", styles))

    subsection_title("5.2 Staged and Curated Validation", story, styles)
    if not retail_validation.empty and "status" in retail_validation.columns:
        story.append(df_to_table(retail_validation.groupby("status").size().reset_index(name="check_count"), styles))
    else:
        story.append(paragraph("Retailrocket validation summary not available.", styles))

    story.append(
        paragraph(
            "Validation checks include missing values, duplicate IDs, schema mismatch, valid event types, "
            "transaction ID consistency, latest item metadata uniqueness, and metadata coverage. The current "
            "reports contain no failing validation checks.",
            styles,
        )
    )
    warning_df = validation_warning_details(retail_validation)
    if warning_df.empty:
        story.append(paragraph("No warning-level staged validation checks are present in the current report.", styles))
    else:
        story.append(
            paragraph(
                "Warning-level checks are documented and do not block the pipeline. Current warnings are related "
                "to duplicate event groups and item metadata coverage for category and availability fields.",
                styles,
            )
        )
        story.append(df_to_table(warning_df, styles, max_rows=6, max_cols=4))

    section_title("6. Data Preparation and EDA Plots", story, styles)
    story.append(
        paragraph(
            "Raw CSV/API data is converted into clean staged Parquet datasets and curated analytical datasets. "
            "EDA artifacts summarize event distribution, item popularity, user activity distribution, and model metrics.",
            styles,
        )
    )
    subsection_title("6.1 Staged Preparation Summary", story, styles)
    story.append(df_to_table(preparation, styles, max_rows=8, max_cols=7))
    subsection_title("6.2 Curated Dataset Summary", story, styles)
    story.append(df_to_table(curated, styles, max_rows=5, max_cols=7))
    story.append(KeepTogether(image_block(PLOTS_DIR / "retailrocket_event_distribution.png", "Retailrocket event distribution.", styles)))
    story.append(KeepTogether(image_block(PLOTS_DIR / "retailrocket_top_items.png", "Top Retailrocket items by popularity score.", styles)))
    story.append(KeepTogether(image_block(PLOTS_DIR / "retailrocket_user_activity_distribution.png", "Retailrocket user activity distribution.", styles)))
    story.append(KeepTogether(image_block(PLOTS_DIR / "retailrocket_model_metrics.png", "Retailrocket model metrics.", styles)))

    section_title("7. SQL Warehouse Schema and Transformation", story, styles)
    story.append(
        paragraph(
            "DuckDB is used as a local analytical warehouse. Staged Parquet files are loaded into staged tables and "
            "transformed into mart and feature tables.",
            styles,
        )
    )
    story.append(df_to_table(retail_warehouse, styles, max_rows=12, max_cols=6))
    subsection_title("7.1 SQL Schema Sample", story, styles)
    story.append(df_to_table(sql_schema, styles, max_rows=18, max_cols=6))

    section_title("8. Feature Engineering Logic", story, styles)
    story.append(
        paragraph(
            "Feature engineering creates user-level, item-level, and user-item interaction features for recommendation.",
            styles,
        )
    )
    story.append(df_to_table(feature_logic, styles, max_rows=14, max_cols=5, column_widths=[1.3 * inch, 1.6 * inch, 1.6 * inch, 1.8 * inch, 1.2 * inch]))

    section_title("9. Feature Store and Feature Metadata", story, styles)
    story.append(
        paragraph(
            "A custom feature registry is implemented to demonstrate feature store concepts. Feature metadata "
            "documents feature views, entities, source paths, column roles, data types, and version.",
            styles,
        )
    )
    subsection_title("9.1 Sample Feature Retrieval Demonstration", story, styles)
    story.append(df_to_table(feature_retrieval, styles, max_rows=8, max_cols=5))
    subsection_title("9.2 Feature Metadata Documentation Sample", story, styles)
    story.append(df_to_table(feature_metadata, styles, max_rows=20, max_cols=7))

    section_title("10. Data Versioning and Lineage with DVC", story, styles)
    story.append(
        paragraph(
            "DVC is used to track large datasets and model artifacts while Git stores source code, reports, and DVC "
            "metadata files. This provides reproducibility without committing heavy files directly to Git.",
            styles,
        )
    )
    story.append(df_to_table(dvc_summary, styles, max_rows=10, max_cols=8))
    subsection_title("10.1 Repository Structure and Dataset Versioning", story, styles)
    story.append(df_to_table(repo_structure, styles, max_rows=15, max_cols=4, column_widths=[1.5 * inch, 2.2 * inch, 1.4 * inch, 1.9 * inch]))

    section_title("11. Model Training and Evaluation", story, styles)
    story.append(
        paragraph(
            "The Retailrocket modeling stage includes two recommenders: an event-weighted global popularity baseline "
            "and a category/content-based filtering model. Event weights are: view = 1, addtocart = 3, "
            "transaction = 5. A time-based train/test split is used.",
            styles,
        )
    )
    story.append(df_to_table(retail_training, styles, max_rows=5, max_cols=12))
    subsection_title("11.2 Content-Based Recommender", story, styles)
    story.append(
        paragraph(
            "A second Retailrocket model was implemented using category/content-based filtering. "
            "It builds user profiles from historical item categories and parent categories, then scores "
            "candidate items using category similarity, availability metadata, conversion signals, and "
            "a popularity prior. This directly addresses the assignment requirement for content-based filtering.",
            styles,
        )
    )
    story.append(df_to_table(retail_content_training, styles, max_rows=5, max_cols=12))

    subsection_title("11.3 Model Comparison", story, styles)
    story.append(df_to_table(retail_model_comparison, styles, max_rows=5, max_cols=10))
    story.append(
        KeepTogether(
            image_block(
                PLOTS_DIR / "retailrocket_model_comparison_metrics.png",
                "Comparison of popularity baseline and content-based recommender metrics.",
                styles,
            )
        )
    )
    if not retail_training.empty:
        row = retail_training.iloc[0]
        story.append(
            paragraph(
                f"Retailrocket performance: HitRate@10 = {row['hit_rate_at_10']:.6f}, "
                f"Precision@10 = {row['precision_at_10']:.6f}, "
                f"Recall@10 = {row['recall_at_10']:.6f}, "
                f"NDCG@10 = {row['ndcg_at_10']:.6f}.",
                styles,
            )
        )
    story.append(
        paragraph(
            "NDCG@10 is ranking-aware: a relevant item at rank 1 receives more credit than one at rank 10.",
            styles,
        )
    )

    section_title("12. MLflow Tracking Evidence", story, styles)
    story.append(
        paragraph(
            "MLflow stores run IDs, parameters, metrics, and artifacts. Screenshots below show successful experiment "
            "tracking for Retailrocket models.",
            styles,
        )
    )
    screenshot_files = [
        ("mlflow_01_experiments_list.png", "MLflow experiments list."),
        ("mlflow_02_retailrocket_runs_table.png", "Retailrocket runs table with metrics."),
        ("mlflow_03_retailrocket_run_overview.png", "Retailrocket run overview."),
        ("mlflow_04_retailrocket_model_metrics.png", "Retailrocket metric details."),
        ("mlflow_06_retailrocket_content_based_run.png", "Retailrocket content-based recommender run in MLflow."),
    ]

    for file_name, caption in screenshot_files:
        story.append(KeepTogether(image_block(SCREENSHOTS_DIR / file_name, caption, styles, width=6.6 * inch)))

    section_title("13. Inference Demo", story, styles)
    story.append(
        paragraph(
            "The inference script loads the model and generates recommendations for default active users, one user, "
            "or custom comma-separated users. Previously seen items are filtered.",
            styles,
        )
    )
    story.append(df_to_table(retail_inference, styles, max_rows=3, max_cols=10))

    section_title("14. Pipeline Orchestration", story, styles)
    story.append(
        paragraph(
            "Prefect orchestrates the Retailrocket-only pipeline. The flow runs batch ingestion, REST or mock API "
            "delta ingestion, raw validation, staged and curated preparation, warehouse loading, feature building, "
            "feature retrieval, model training, inference, comparison, and report generation.",
            styles,
        )
    )
    story.append(df_to_table(retail_orchestration[["step_order", "step_name", "status", "duration_seconds"]], styles, max_rows=20, max_cols=4))

    section_title("15. Reproducibility Workflow", story, styles)
    story.append(
        bullet_list(
            [
                "Activate environment: conda activate recomart",
                "Check version state: git status and dvc status",
                "Restore tracked artifacts if remote is configured: dvc pull",
                "Run Retailrocket flow: python -m orchestration.retailrocket_pipeline",
                "Run mock API ingestion: RECOMART_CATALOG_API_MOCK_MODE=true python -m src.ingestion.ingest_retailrocket_catalog_api",
                "Run real API ingestion: RECOMART_CATALOG_API_MOCK_MODE=false RECOMART_CATALOG_API_BASE_URL=<teammate-api-base-url> RECOMART_CATALOG_API_ENDPOINT=/items RECOMART_CATALOG_API_PAGE_SIZE=50 python -m src.ingestion.ingest_retailrocket_catalog_api",
                "Schedule REST ingestion every 30 minutes: prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow --name retailrocket-api-every-30-minutes --interval 1800 --pool default-agent-pool",
                "Open MLflow UI using the tracking URI from Python: mlflow ui --backend-store-uri \"$TRACKING_URI\" --port 5001",
            ],
            styles,
        )
    )

    section_title("16. Operational Notes", story, styles)
    story.append(
        bullet_list(
            [
                "Current Retailrocket models are lightweight and suitable for the assignment scope.",
                "The content-based model is the main assignment model; popularity is retained as a benchmark.",
                "The API ingestion flow supports local mock mode and environment-based teammate API configuration.",
                "The real API serves new rows without repetition; the ingestion client records API count, total rows, unread rows, and success metadata.",
                "Generated heavy data and model files are DVC-tracked rather than committed directly to Git.",
                "The Prefect API ingestion deployment command documents the 30-minute periodic metadata refresh.",
            ],
            styles,
        )
    )

    section_title("17. Conclusion", story, styles)
    story.append(
        paragraph(
            "The project implements a complete and reproducible ML data management pipeline for recommendation "
            "systems using Python, DuckDB, Parquet, DVC, MLflow, and Prefect. It satisfies the required stages "
            "from ingestion through orchestration and produces source code, reports, screenshots, and a final PDF.",
            styles,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="RecoMart Assignment Report",
        author="RecoMart Data Platform Team",
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    print(f"PDF report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
