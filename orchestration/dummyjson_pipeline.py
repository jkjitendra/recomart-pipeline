import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports/orchestration_dummyjson_pipeline_summary.csv"


PIPELINE_STEPS = [
    {
        "step_order": 1,
        "step_name": "ingest_dummyjson_api",
        "module": "src.ingestion.fetch_dummyjson",
        "description": "Fetch products, users, and carts from DummyJSON API into raw JSON files.",
    },
    {
        "step_order": 2,
        "step_name": "inspect_dummyjson_raw",
        "module": "src.ingestion.inspect_dummyjson_raw",
        "description": "Inspect raw JSON files and generate raw summary report.",
    },
    {
        "step_order": 3,
        "step_name": "validate_dummyjson_raw",
        "module": "src.validation.validate_dummyjson_raw",
        "description": "Validate raw DummyJSON files for schema, missing fields, duplicates, and value checks.",
    },
    {
        "step_order": 4,
        "step_name": "prepare_dummyjson_staged",
        "module": "src.preparation.prepare_dummyjson",
        "description": "Convert raw JSON files into staged Parquet tables.",
    },
    {
        "step_order": 5,
        "step_name": "validate_dummyjson_staged",
        "module": "src.validation.validate_dummyjson_staged",
        "description": "Validate staged Parquet tables and referential integrity.",
    },
    {
        "step_order": 6,
        "step_name": "load_dummyjson_to_duckdb",
        "module": "src.transformation.load_dummyjson_to_duckdb",
        "description": "Load staged Parquet files into DuckDB staged tables and mart views.",
    },
    {
        "step_order": 7,
        "step_name": "build_dummyjson_features",
        "module": "src.transformation.build_dummyjson_features",
        "description": "Build user, item, and interaction feature tables.",
    },
    {
        "step_order": 8,
        "step_name": "retrieve_dummyjson_features",
        "module": "src.feature_store.retrieve_dummyjson_features",
        "description": "Run custom feature registry summary and feature retrieval demo.",
    },
    {
        "step_order": 9,
        "step_name": "train_popularity_recommender",
        "module": "src.training.train_popularity_recommender",
        "description": "Train popularity recommender baseline and log MLflow metrics.",
    },
    {
        "step_order": 10,
        "step_name": "run_popularity_inference",
        "module": "src.serving.recommend_popularity",
        "description": "Generate demo recommendations using trained popularity model.",
    },
]


def tail_text(text: str, max_chars: int = 1200) -> str:
    """
    Keeps only the last part of stdout/stderr so the orchestration report
    remains readable.
    """
    if not text:
        return ""

    return text[-max_chars:]


@task(name="run_pipeline_step", retries=0)
def run_pipeline_step(step: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    command = [sys.executable, "-m", step["module"]]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    ended_at = datetime.now().isoformat(timespec="seconds")
    duration_seconds = round(time.time() - start_time, 3)

    status = "PASS" if result.returncode == 0 else "FAIL"

    row = {
        "step_order": step["step_order"],
        "step_name": step["step_name"],
        "module": step["module"],
        "description": step["description"],
        "command": " ".join(command),
        "status": status,
        "return_code": result.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "stdout_tail": tail_text(result.stdout),
        "stderr_tail": tail_text(result.stderr),
    }

    print(f"\n===== {step['step_order']}. {step['step_name']} =====")
    print(f"Command: {' '.join(command)}")
    print(f"Status: {status}")
    print(f"Duration seconds: {duration_seconds}")

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    return row


@task(name="write_orchestration_report")
def write_orchestration_report(rows: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"\nOrchestration summary saved: {REPORT_PATH}")
    print(report_df[["step_order", "step_name", "status", "duration_seconds"]].to_string(index=False))


@flow(name="dummyjson-recomart-pipeline")
def dummyjson_pipeline() -> None:
    rows = []

    for step in PIPELINE_STEPS:
        row = run_pipeline_step(step)
        rows.append(row)

        if row["status"] != "PASS":
            write_orchestration_report(rows)
            raise RuntimeError(
                f"Pipeline failed at step {row['step_order']}: {row['step_name']}"
            )

    write_orchestration_report(rows)


if __name__ == "__main__":
    dummyjson_pipeline()