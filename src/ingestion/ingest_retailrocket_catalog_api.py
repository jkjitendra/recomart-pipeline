from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data/raw/source=retailrocket_api"
LOG_PATH = PROJECT_ROOT / "logs/retailrocket_catalog_api_ingestion.log"
SUMMARY_PATH = PROJECT_ROOT / "reports/retailrocket_api_ingestion_summary.csv"
DEFAULT_STATE_PATH = RAW_ROOT / "_state/item_properties_delta_state.json"
MOCK_SOURCE_PATH = PROJECT_ROOT / "data/external/retailrocket/item_properties_part2.csv"

SOURCE_SYSTEM = "retailrocket_api"
DATA_TYPE = "item_properties_delta"


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    endpoint: str
    timeout_sec: float
    page_size: int
    auth_token: str | None
    state_file: Path
    mock_mode: bool
    retries: int = 3
    backoff_sec: float = 1.5


def configure_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("recomart.retailrocket_catalog_api_ingestion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    path = Path(value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config() -> ApiConfig:
    page_size = int(os.getenv("RECOMART_CATALOG_API_PAGE_SIZE", "10"))

    if page_size <= 0:
        raise ValueError("RECOMART_CATALOG_API_PAGE_SIZE must be positive.")

    return ApiConfig(
        base_url=os.getenv("RECOMART_CATALOG_API_BASE_URL", "http://127.0.0.1:8000"),
        endpoint=os.getenv("RECOMART_CATALOG_API_ENDPOINT", "/api/item-properties"),
        timeout_sec=float(os.getenv("RECOMART_CATALOG_API_TIMEOUT_SEC", "30")),
        page_size=page_size,
        auth_token=os.getenv("RECOMART_CATALOG_API_AUTH_TOKEN") or None,
        state_file=env_path("RECOMART_CATALOG_API_STATE_FILE", DEFAULT_STATE_PATH),
        mock_mode=parse_bool(os.getenv("RECOMART_CATALOG_API_MOCK_MODE"), default=True),
    )


def ingestion_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def landing_dir(run_timestamp: str) -> Path:
    return RAW_ROOT / f"type={DATA_TYPE}" / f"ingestion_timestamp={run_timestamp}"


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {
            "cursor": None,
            "records_ingested_total": 0,
            "last_ingestion_timestamp": None,
            "last_api_count": None,
            "last_api_total_rows": None,
            "last_api_unread_rows": None,
            "last_api_success": None,
            "updated_at": None,
        }

    with state_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_state(
    *,
    state_file: Path,
    previous_state: dict[str, Any],
    next_cursor: str | None,
    run_timestamp: str,
    records_ingested: int,
    payload_metadata: dict[str, Any],
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)

    updated_state = {
        "cursor": next_cursor,
        "records_ingested_total": int(previous_state.get("records_ingested_total", 0)) + records_ingested,
        "last_ingestion_timestamp": run_timestamp,
        "last_api_count": payload_metadata.get("api_count"),
        "last_api_total_rows": payload_metadata.get("total_rows"),
        "last_api_unread_rows": payload_metadata.get("unread_rows"),
        "last_api_success": payload_metadata.get("success"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    state_file.write_text(json.dumps(updated_state, indent=2), encoding="utf-8")


def normalize_endpoint(base_url: str, endpoint: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def request_api_page(
    *,
    config: ApiConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    url = normalize_endpoint(config.base_url, config.endpoint)
    headers = {}

    if config.auth_token:
        headers["Authorization"] = f"Bearer {config.auth_token}"

    params = {"count": config.page_size}

    for attempt in range(1, config.retries + 1):
        try:
            logger.info(
                "Fetching Retailrocket API page attempt=%s url=%s count=%s timeout=%s",
                attempt,
                url,
                config.page_size,
                config.timeout_sec,
            )

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=config.timeout_sec,
            )
            response.raise_for_status()

            return response.json()
        except requests.RequestException as exc:
            logger.warning("API request failed attempt=%s error=%s", attempt, exc)
            if attempt == config.retries:
                logger.exception("API request failed after all retries.")
                raise
            time.sleep(config.backoff_sec * attempt)

    raise RuntimeError("Unexpected API retry loop exit.")


def parse_cursor(cursor: str | None) -> int:
    if cursor in (None, ""):
        return 0
    return max(int(cursor), 0)


def coerce_mock_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "timestamp": int(row["timestamp"]),
        "itemid": int(row["itemid"]),
        "property": str(row["property"]),
        "value": str(row["value"]),
    }


def read_mock_page(limit: int, cursor: str | None) -> dict[str, Any]:
    if not MOCK_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Mock source file not found: {MOCK_SOURCE_PATH}")

    offset = parse_cursor(cursor)
    records: list[dict[str, Any]] = []
    has_more = False

    with MOCK_SOURCE_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader):
            if index < offset:
                continue

            if len(records) < limit:
                records.append(coerce_mock_record(row))
                continue

            has_more = True
            break

    next_cursor = str(offset + len(records)) if records else str(offset)

    return {
        "records": records,
        "next_cursor": next_cursor,
        "has_more": has_more,
    }


def fetch_page(
    *,
    config: ApiConfig,
    cursor: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    if config.mock_mode:
        logger.info(
            "Using mock mode source=%s limit=%s cursor=%s",
            MOCK_SOURCE_PATH,
            config.page_size,
            cursor,
        )
        return read_mock_page(config.page_size, cursor)

    return request_api_page(config=config, logger=logger)


def detect_api_contract(payload: dict[str, Any]) -> str:
    if "data" in payload:
        return "teammate_data_contract"

    if "records" in payload:
        return "mock_records_contract"

    raise ValueError("API response must contain either 'data' or 'records'.")


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    api_contract = detect_api_contract(payload)

    if api_contract == "teammate_data_contract":
        if not isinstance(payload.get("data"), list):
            raise ValueError("Teammate API response field 'data' must be a list.")

        success = bool(payload.get("success", False))
        if not success:
            raise ValueError("Teammate API response reported success=false.")

        api_count = payload.get("count")
        if api_count is not None and int(api_count) != len(payload["data"]):
            raise ValueError(
                f"Teammate API count mismatch: count={api_count}, data_rows={len(payload['data'])}"
            )

        return {
            "api_contract": api_contract,
            "records": payload["data"],
            "next_cursor": None,
            "has_more": None,
            "api_count": int(api_count) if api_count is not None else len(payload["data"]),
            "total_rows": payload.get("total_rows"),
            "unread_rows": payload.get("unread_rows"),
            "success": success,
        }

    if not isinstance(payload.get("records"), list):
        raise ValueError("Mock/internal API response field 'records' must be a list.")

    if "next_cursor" not in payload:
        raise ValueError("Mock/internal API response must contain 'next_cursor'.")

    if "has_more" not in payload:
        raise ValueError("Mock/internal API response must contain 'has_more'.")

    return {
        "api_contract": api_contract,
        "records": payload["records"],
        "next_cursor": payload.get("next_cursor"),
        "has_more": payload.get("has_more"),
        "api_count": len(payload["records"]),
        "total_rows": payload.get("total_rows"),
        "unread_rows": payload.get("unread_rows"),
        "success": payload.get("success", True),
    }


def normalize_records(records: list[dict[str, Any]], run_timestamp: str) -> pd.DataFrame:
    expected_columns = ["timestamp", "itemid", "property", "value"]

    rows = []
    for record in records:
        rows.append(
            {
                "timestamp": int(record["timestamp"]),
                "itemid": int(record["itemid"]),
                "property": str(record["property"]),
                "value": str(record["value"]),
                "source_system": SOURCE_SYSTEM,
                "ingestion_timestamp": run_timestamp,
            }
        )

    if not rows:
        return pd.DataFrame(columns=[*expected_columns, "source_system", "ingestion_timestamp"])

    return pd.DataFrame(rows)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_landing_files(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    normalized_df: pd.DataFrame,
    run_timestamp: str,
    config: ApiConfig,
    state_before: dict[str, Any],
    payload_metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_response_path = output_dir / "raw_response.json"
    normalized_json_path = output_dir / "item_properties_delta.json"
    normalized_csv_path = output_dir / "item_properties_delta.csv"
    normalized_parquet_path = output_dir / "item_properties_delta.parquet"
    metadata_path = output_dir / "_metadata.json"

    raw_response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    normalized_json_path.write_text(
        normalized_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    normalized_df.to_csv(normalized_csv_path, index=False)
    normalized_df.to_parquet(normalized_parquet_path, index=False)
    normalized_csv_checksum = file_sha256(normalized_csv_path)

    metadata = {
        "source_system": SOURCE_SYSTEM,
        "data_type": DATA_TYPE,
        "ingestion_timestamp": run_timestamp,
        "row_count": int(len(normalized_df)),
        "request_count": config.page_size,
        "page_size": config.page_size,
        "mock_mode": config.mock_mode,
        "api_contract": payload_metadata.get("api_contract"),
        "total_rows": payload_metadata.get("total_rows"),
        "unread_rows": payload_metadata.get("unread_rows"),
        "api_count": payload_metadata.get("api_count"),
        "api_success": payload_metadata.get("success"),
        "checksum_sha256": normalized_csv_checksum,
        "base_url": config.base_url,
        "endpoint": config.endpoint,
        "state_file": str(config.state_file.relative_to(PROJECT_ROOT))
        if config.state_file.is_relative_to(PROJECT_ROOT)
        else str(config.state_file),
        "cursor_before": state_before.get("cursor"),
        "next_cursor": payload_metadata.get("next_cursor"),
        "has_more": payload_metadata.get("has_more"),
        "raw_response_file": str(raw_response_path.relative_to(PROJECT_ROOT)),
        "normalized_json_file": str(normalized_json_path.relative_to(PROJECT_ROOT)),
        "normalized_csv_file": str(normalized_csv_path.relative_to(PROJECT_ROOT)),
        "normalized_parquet_file": str(normalized_parquet_path.relative_to(PROJECT_ROOT)),
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "mock_source_file": str(MOCK_SOURCE_PATH.relative_to(PROJECT_ROOT)) if config.mock_mode else "",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "raw_response": raw_response_path,
        "normalized_json": normalized_json_path,
        "normalized_csv": normalized_csv_path,
        "normalized_parquet": normalized_parquet_path,
        "metadata": metadata_path,
    }


def path_for_summary(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def build_summary_row(
    *,
    run_timestamp: str,
    config: ApiConfig,
    state_before: dict[str, Any],
    payload: dict[str, Any],
    payload_metadata: dict[str, Any],
    normalized_df: pd.DataFrame,
    output_dir: Path,
    files: dict[str, Path],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "source_system": SOURCE_SYSTEM,
        "data_type": DATA_TYPE,
        "ingestion_timestamp": run_timestamp,
        "landing_path": path_for_summary(output_dir),
        "raw_response_file": path_for_summary(files["raw_response"]) if files else "",
        "normalized_json_file": path_for_summary(files["normalized_json"]) if files else "",
        "normalized_csv_file": path_for_summary(files["normalized_csv"]) if files else "",
        "normalized_parquet_file": path_for_summary(files["normalized_parquet"]) if files else "",
        "metadata_file": path_for_summary(files["metadata"]) if files else "",
        "state_file": path_for_summary(config.state_file),
        "mock_mode": config.mock_mode,
        "api_contract": payload_metadata.get("api_contract") if payload_metadata else "",
        "page_size": config.page_size,
        "cursor_before": state_before.get("cursor"),
        "next_cursor": payload_metadata.get("next_cursor") if payload_metadata else None,
        "has_more": payload_metadata.get("has_more") if payload_metadata else None,
        "row_count": int(len(normalized_df)),
        "total_rows": payload_metadata.get("total_rows") if payload_metadata else None,
        "unread_rows": payload_metadata.get("unread_rows") if payload_metadata else None,
        "api_success": payload_metadata.get("success") if payload_metadata else None,
        "status": status,
        "error": error,
    }


def main() -> None:
    logger = configure_logger()
    config = load_config()
    run_timestamp = ingestion_timestamp()
    output_dir = landing_dir(run_timestamp)
    state_before = load_state(config.state_file)

    logger.info(
        "Starting Retailrocket catalog API ingestion ingestion_timestamp=%s mock_mode=%s state_file=%s",
        run_timestamp,
        config.mock_mode,
        config.state_file,
    )

    payload: dict[str, Any] = {}
    payload_metadata: dict[str, Any] = {}
    normalized_df = pd.DataFrame()
    files: dict[str, Path] = {}

    try:
        payload = fetch_page(config=config, cursor=state_before.get("cursor"), logger=logger)
        payload_metadata = validate_response(payload)
        normalized_df = normalize_records(payload_metadata["records"], run_timestamp)
        files = write_landing_files(
            output_dir=output_dir,
            payload=payload,
            normalized_df=normalized_df,
            run_timestamp=run_timestamp,
            config=config,
            state_before=state_before,
            payload_metadata=payload_metadata,
        )
        save_state(
            state_file=config.state_file,
            previous_state=state_before,
            next_cursor=str(payload_metadata.get("next_cursor"))
            if payload_metadata.get("next_cursor") is not None
            else state_before.get("cursor"),
            run_timestamp=run_timestamp,
            records_ingested=len(normalized_df),
            payload_metadata=payload_metadata,
        )

        summary_row = build_summary_row(
            run_timestamp=run_timestamp,
            config=config,
            state_before=state_before,
            payload=payload,
            payload_metadata=payload_metadata,
            normalized_df=normalized_df,
            output_dir=output_dir,
            files=files,
            status="PASS",
        )
        logger.info(
            "Retailrocket catalog API ingestion succeeded rows=%s next_cursor=%s has_more=%s landing_path=%s",
            len(normalized_df),
            payload_metadata.get("next_cursor"),
            payload_metadata.get("has_more"),
            output_dir,
        )
    except Exception as exc:
        logger.exception("Retailrocket catalog API ingestion failed.")
        summary_row = build_summary_row(
            run_timestamp=run_timestamp,
            config=config,
            state_before=state_before,
            payload=payload,
            payload_metadata=payload_metadata,
            normalized_df=normalized_df,
            output_dir=output_dir,
            files=files,
            status="FAIL",
            error=str(exc),
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("Retailrocket catalog API ingestion completed.")
    print(f"Ingestion timestamp: {run_timestamp}")
    print(f"Mock mode: {config.mock_mode}")
    print(f"Summary saved: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Log saved: {LOG_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(summary_df.to_string(index=False))

    if summary_row["status"] != "PASS":
        raise RuntimeError(f"Retailrocket catalog API ingestion failed: {summary_row['error']}")


if __name__ == "__main__":
    main()
