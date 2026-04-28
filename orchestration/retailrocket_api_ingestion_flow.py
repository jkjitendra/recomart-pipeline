from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import flow, task


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports/orchestration_retailrocket_api_ingestion_summary.csv"


def tail_text(text: str, max_lines: int = 30) -> str:
    if not text:
        return ""

    return "\n".join(text.splitlines()[-max_lines:])


@task
def run_api_ingestion_command() -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("RECOMART_CATALOG_API_MOCK_MODE", "false")
    env.setdefault(
        "RECOMART_CATALOG_API_BASE_URL",
        "https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud",
    )
    env.setdefault("RECOMART_CATALOG_API_ENDPOINT", "/items")
    env.setdefault("RECOMART_CATALOG_API_PAGE_SIZE", "50")

    command = [
        sys.executable,
        "-m",
        "src.ingestion.ingest_retailrocket_catalog_api",
    ]

    started_at = datetime.now().replace(microsecond=0).isoformat()
    start_time = time.time()

    completed_process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    ended_at = datetime.now().replace(microsecond=0).isoformat()
    duration_seconds = round(time.time() - start_time, 3)
    status = "PASS" if completed_process.returncode == 0 else "FAIL"

    result = {
        "step_order": 1,
        "step_name": "ingest_retailrocket_catalog_api_delta",
        "module": "src.ingestion.ingest_retailrocket_catalog_api",
        "command": (
            "RECOMART_CATALOG_API_MOCK_MODE=${RECOMART_CATALOG_API_MOCK_MODE:-false} "
            "RECOMART_CATALOG_API_BASE_URL=${RECOMART_CATALOG_API_BASE_URL:-https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud} "
            "RECOMART_CATALOG_API_ENDPOINT=${RECOMART_CATALOG_API_ENDPOINT:-/items} "
            "RECOMART_CATALOG_API_PAGE_SIZE=${RECOMART_CATALOG_API_PAGE_SIZE:-50} "
            "python -m src.ingestion.ingest_retailrocket_catalog_api"
        ),
        "description": "Scheduled Retailrocket catalog metadata delta ingestion.",
        "status": status,
        "return_code": completed_process.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "stdout_tail": tail_text(completed_process.stdout),
        "stderr_tail": tail_text(completed_process.stderr),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(REPORT_PATH, index=False)

    print(f"API ingestion orchestration summary saved: {REPORT_PATH}")
    print(pd.DataFrame([result])[["step_name", "status", "return_code", "duration_seconds"]].to_string(index=False))

    if completed_process.stdout:
        print("\nSTDOUT tail:")
        print(result["stdout_tail"])

    if completed_process.stderr:
        print("\nSTDERR tail:")
        print(result["stderr_tail"])

    if completed_process.returncode != 0:
        raise RuntimeError("Retailrocket API ingestion flow failed.")

    return result


@flow(name="retailrocket-api-ingestion-flow")
def retailrocket_api_ingestion_flow() -> None:
    """
    Runs one Retailrocket catalog metadata delta ingestion.

    The teammate-hosted API is the default source. Set RECOMART_CATALOG_API_MOCK_MODE=true
    only when a reproducible local mock run is needed. The ingestion client sends
    RECOMART_CATALOG_API_PAGE_SIZE as the count query parameter.

    Prefect 3 local deployment example:

    prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
      --name retailrocket-api-every-30-minutes \
      --interval 1800 \
      --pool default-agent-pool
    """
    run_api_ingestion_command()


if __name__ == "__main__":
    retailrocket_api_ingestion_flow()
