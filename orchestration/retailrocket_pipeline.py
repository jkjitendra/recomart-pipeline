from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports/orchestration_retailrocket_pipeline_summary.csv"

STD_TAIL_LINES = 30


PIPELINE_STEPS: list[dict[str, Any]] = [
    {
        "step_order": 1,
        "step_name": "inspect_retailrocket_external",
        "module": "src.ingestion.inspect_retailrocket_external",
        "description": "Inspect Retailrocket external CSV source archive and write source summaries.",
    },
    {
        "step_order": 2,
        "step_name": "ingest_retailrocket_batch",
        "module": "src.ingestion.ingest_retailrocket_batch",
        "description": "Land Retailrocket batch CSV files into timestamped raw data lake partitions.",
    },
    {
        "step_order": 3,
        "step_name": "ingest_retailrocket_catalog_api_delta",
        "module": "src.ingestion.ingest_retailrocket_catalog_api",
        "description": "Land Retailrocket catalog metadata deltas from the teammate REST API or reproducible mock API.",
        "env_defaults": {
            "RECOMART_CATALOG_API_MOCK_MODE": "false",
            "RECOMART_CATALOG_API_BASE_URL": "https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud",
            "RECOMART_CATALOG_API_ENDPOINT": "/items",
            "RECOMART_CATALOG_API_PAGE_SIZE": "50",
        },
        "command_prefix": (
            "RECOMART_CATALOG_API_MOCK_MODE=${RECOMART_CATALOG_API_MOCK_MODE:-false} "
            "RECOMART_CATALOG_API_BASE_URL=${RECOMART_CATALOG_API_BASE_URL:-https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud} "
            "RECOMART_CATALOG_API_ENDPOINT=${RECOMART_CATALOG_API_ENDPOINT:-/items} "
            "RECOMART_CATALOG_API_PAGE_SIZE=${RECOMART_CATALOG_API_PAGE_SIZE:-50}"
        ),
    },
    {
        "step_order": 4,
        "step_name": "validate_retailrocket_raw",
        "module": "src.validation.validate_retailrocket_raw",
        "description": "Validate raw Retailrocket batch and API landing partitions.",
    },
    {
        "step_order": 5,
        "step_name": "prepare_retailrocket_staged_and_curated",
        "module": "src.preparation.prepare_retailrocket",
        "description": "Prepare staged Parquet outputs and curated analytical datasets from raw data.",
    },
    {
        "step_order": 6,
        "step_name": "validate_retailrocket_staged_and_curated",
        "module": "src.validation.validate_retailrocket_staged",
        "description": "Validate staged and curated Retailrocket datasets.",
    },
    {
        "step_order": 7,
        "step_name": "load_retailrocket_to_duckdb",
        "module": "src.transformation.load_retailrocket_to_duckdb",
        "description": "Load staged and curated Retailrocket data into the DuckDB warehouse.",
    },
    {
        "step_order": 8,
        "step_name": "build_retailrocket_features",
        "module": "src.transformation.build_retailrocket_features",
        "description": "Build Retailrocket user, item, and user-item feature tables.",
    },
    {
        "step_order": 9,
        "step_name": "retrieve_retailrocket_features",
        "module": "src.feature_store.retrieve_retailrocket_features",
        "description": "Run Retailrocket feature registry and retrieval demo.",
    },
    {
        "step_order": 10,
        "step_name": "train_retailrocket_popularity_recommender",
        "module": "src.training.train_retailrocket_popularity_recommender",
        "description": "Train Retailrocket popularity baseline and log MLflow metrics.",
    },
    {
        "step_order": 11,
        "step_name": "train_retailrocket_content_recommender",
        "module": "src.training.train_retailrocket_content_recommender",
        "description": "Train Retailrocket content-based recommender and log MLflow metrics.",
    },
    {
        "step_order": 12,
        "step_name": "run_retailrocket_popularity_inference",
        "module": "src.serving.recommend_retailrocket",
        "args": ["--model", "popularity", "--top-k", "5"],
        "description": "Generate inference examples using the Retailrocket popularity model.",
    },
    {
        "step_order": 13,
        "step_name": "run_retailrocket_content_based_inference",
        "module": "src.serving.recommend_retailrocket",
        "args": ["--model", "content_based", "--top-k", "5"],
        "description": "Generate inference examples using the Retailrocket content-based model.",
    },
    {
        "step_order": 14,
        "step_name": "generate_model_comparison",
        "module": "src.reporting.generate_model_comparison",
        "description": "Generate Retailrocket model comparison CSV and plot.",
    },
    {
        "step_order": 15,
        "step_name": "generate_assignment_evidence",
        "module": "src.reporting.generate_assignment_evidence",
        "description": "Generate assignment evidence summaries and EDA plots where available.",
    },
    {
        "step_order": 16,
        "step_name": "generate_final_report",
        "module": "src.reporting.generate_final_report",
        "description": "Generate the markdown final project report and reproducibility commands.",
    },
    {
        "step_order": 17,
        "step_name": "generate_assignment_pdf",
        "module": "src.reporting.generate_assignment_pdf",
        "description": "Generate the RecoMart assignment PDF report from current evidence.",
    },
]


def tail_text(text: str, max_lines: int = STD_TAIL_LINES) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def build_command(step: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        step["module"],
        *step.get("args", []),
    ]


def display_command(step: dict[str, Any]) -> str:
    command = ["python", "-m", step["module"], *step.get("args", [])]
    command_text = " ".join(command)
    command_prefix = step.get("command_prefix")

    if command_prefix:
        return f"{command_prefix} {command_text}"

    return command_text


def build_environment(step: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()

    for key, value in step.get("env_defaults", {}).items():
        env.setdefault(key, value)

    return env


@task
def run_pipeline_step(step: dict[str, Any]) -> dict[str, Any]:
    command = build_command(step)
    command_text = display_command(step)
    started_at = datetime.now().replace(microsecond=0).isoformat()
    start_time = time.time()

    print(f"\n===== {step['step_order']}. {step['step_name']} =====")
    print(f"Description: {step['description']}")
    print(f"Command: {command_text}")

    completed_process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=build_environment(step),
        capture_output=True,
        text=True,
    )

    ended_at = datetime.now().replace(microsecond=0).isoformat()
    duration_seconds = round(time.time() - start_time, 3)
    status = "PASS" if completed_process.returncode == 0 else "FAIL"

    stdout_tail = tail_text(completed_process.stdout)
    stderr_tail = tail_text(completed_process.stderr)

    print(f"Status: {status}")
    print(f"Return code: {completed_process.returncode}")
    print(f"Duration seconds: {duration_seconds}")

    if stdout_tail:
        print("\nSTDOUT tail:")
        print(stdout_tail)

    if stderr_tail:
        print("\nSTDERR tail:")
        print(stderr_tail)

    return {
        "step_order": step["step_order"],
        "step_name": step["step_name"],
        "module": step["module"],
        "command": command_text,
        "description": step["description"],
        "status": status,
        "return_code": completed_process.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@task
def write_orchestration_report(results: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"\nOrchestration summary saved: {REPORT_PATH}")
    print()
    print(
        report_df[
            [
                "step_order",
                "step_name",
                "status",
                "return_code",
                "duration_seconds",
            ]
        ].to_string(index=False)
    )


@flow(name="retailrocket-recomart-pipeline")
def retailrocket_recomart_pipeline() -> None:
    results = []

    for step in PIPELINE_STEPS:
        result = run_pipeline_step(step)
        results.append(result)

    write_orchestration_report(results)

    failed_steps = [
        result
        for result in results
        if result["status"] != "PASS"
    ]

    if failed_steps:
        failed_names = ", ".join(result["step_name"] for result in failed_steps)
        raise RuntimeError(f"Retailrocket Prefect pipeline failed steps: {failed_names}")


if __name__ == "__main__":
    retailrocket_recomart_pipeline()
