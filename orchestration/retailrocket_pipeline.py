import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task


REPORT_PATH = Path("reports/orchestration_retailrocket_pipeline_summary.csv")


PIPELINE_STEPS = [
    {
        "step_order": 1,
        "step_name": "inspect_retailrocket_external",
        "module": "src.ingestion.inspect_retailrocket_external",
        "description": "Inspect Retailrocket external CSV files and generate summary reports.",
    },
    {
        "step_order": 2,
        "step_name": "prepare_retailrocket_staged",
        "module": "src.preparation.prepare_retailrocket",
        "description": "Convert Retailrocket external CSV files into staged Parquet files.",
    },
    {
        "step_order": 3,
        "step_name": "validate_retailrocket_staged",
        "module": "src.validation.validate_retailrocket_staged",
        "description": "Validate Retailrocket staged Parquet files.",
    },
    {
        "step_order": 4,
        "step_name": "load_retailrocket_to_duckdb",
        "module": "src.transformation.load_retailrocket_to_duckdb",
        "description": "Load Retailrocket staged data into DuckDB warehouse and mart tables.",
    },
    {
        "step_order": 5,
        "step_name": "build_retailrocket_features",
        "module": "src.transformation.build_retailrocket_features",
        "description": "Build Retailrocket user, item, and user-item feature tables.",
    },
    {
        "step_order": 6,
        "step_name": "train_retailrocket_popularity_recommender",
        "module": "src.training.train_retailrocket_popularity_recommender",
        "description": "Train Retailrocket event-weighted popularity recommender and log MLflow metrics.",
    },
    {
        "step_order": 7,
        "step_name": "run_retailrocket_inference",
        "module": "src.serving.recommend_retailrocket",
        "description": "Generate Retailrocket demo recommendations using trained model.",
    },
]


def tail_text(text: str, max_lines: int = 25) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


@task
def run_pipeline_step(step: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "-m", step["module"]]

    started_at = datetime.now().replace(microsecond=0).isoformat()
    start_time = time.time()

    print(f"\n===== {step['step_order']}. {step['step_name']} =====")
    print(f"Command: {' '.join(command)}")

    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    ended_at = datetime.now().replace(microsecond=0).isoformat()
    duration_seconds = round(time.time() - start_time, 3)

    status = "PASS" if completed_process.returncode == 0 else "FAIL"

    print(f"Status: {status}")
    print(f"Duration seconds: {duration_seconds}")

    if completed_process.stdout:
        print("\nSTDOUT:")
        print(tail_text(completed_process.stdout, max_lines=40))

    if completed_process.stderr:
        print("\nSTDERR:")
        print(tail_text(completed_process.stderr, max_lines=40))

    result = {
        "step_order": step["step_order"],
        "step_name": step["step_name"],
        "module": step["module"],
        "description": step["description"],
        "command": " ".join(command),
        "status": status,
        "return_code": completed_process.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "stdout_tail": tail_text(completed_process.stdout, max_lines=30),
        "stderr_tail": tail_text(completed_process.stderr, max_lines=30),
    }

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {step['step_name']} "
            f"with return code {completed_process.returncode}"
        )

    return result


@task
def write_orchestration_report(results: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"\nOrchestration summary saved: {REPORT_PATH.resolve()}")
    print()
    print(
        report_df[
            [
                "step_order",
                "step_name",
                "status",
                "duration_seconds",
            ]
        ].to_string(index=False)
    )


@flow(name="retailrocket-recommart-pipeline")
def retailrocket_recommart_pipeline() -> None:
    results = []

    for step in PIPELINE_STEPS:
        result = run_pipeline_step(step)
        results.append(result)

    write_orchestration_report(results)


if __name__ == "__main__":
    retailrocket_recommart_pipeline()