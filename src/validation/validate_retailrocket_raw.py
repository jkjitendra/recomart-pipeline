from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BATCH_DIR = PROJECT_ROOT / "data/raw/source=retailrocket_batch"
RAW_API_DIR = PROJECT_ROOT / "data/raw/source=retailrocket_api"
REPORT_PATH = PROJECT_ROOT / "reports/data_quality/retailrocket_raw_validation_report.csv"


BATCH_TYPES = {
    "events": "events.csv",
    "item_properties_part1": "item_properties_part1.csv",
    "category_tree": "category_tree.csv",
}

API_TYPE = "item_properties_delta"


def display_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def add_result(
    rows: list[dict[str, Any]],
    dataset_name: str,
    check_name: str,
    status: str,
    details: str,
) -> None:
    rows.append(
        {
            "dataset_name": dataset_name,
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def latest_ingestion_dir(raw_source_dir: Path, data_type: str) -> Path:
    type_dir = raw_source_dir / f"type={data_type}"
    if not type_dir.exists():
        raise FileNotFoundError(f"Missing raw type directory: {display_path(type_dir)}")

    candidates = sorted(
        path for path in type_dir.iterdir()
        if path.is_dir() and path.name.startswith("ingestion_timestamp=")
    )

    if not candidates:
        raise FileNotFoundError(f"No ingestion_timestamp partitions found in {display_path(type_dir)}")

    return candidates[-1]


def all_ingestion_dirs(raw_source_dir: Path, data_type: str) -> list[Path]:
    type_dir = raw_source_dir / f"type={data_type}"
    if not type_dir.exists():
        raise FileNotFoundError(f"Missing raw type directory: {display_path(type_dir)}")

    candidates = sorted(
        path for path in type_dir.iterdir()
        if path.is_dir() and path.name.startswith("ingestion_timestamp=")
    )

    if not candidates:
        raise FileNotFoundError(f"No ingestion_timestamp partitions found in {display_path(type_dir)}")

    return candidates


def read_metadata(ingestion_dir: Path) -> dict[str, Any]:
    metadata_path = ingestion_dir / "_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata sidecar: {display_path(metadata_path)}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        row_count = sum(1 for _ in reader)
    return max(row_count - 1, 0)


def validate_batch_landing(rows: list[dict[str, Any]]) -> None:
    for data_type, file_name in BATCH_TYPES.items():
        dataset_name = f"raw_batch_{data_type}"

        try:
            ingestion_dir = latest_ingestion_dir(RAW_BATCH_DIR, data_type)
            metadata = read_metadata(ingestion_dir)
            raw_file = ingestion_dir / file_name

            if raw_file.exists():
                add_result(rows, dataset_name, "raw_file_exists", "PASS", display_path(raw_file))
            else:
                add_result(rows, dataset_name, "raw_file_exists", "FAIL", display_path(raw_file))
                continue

            partition_name = ingestion_dir.name
            if partition_name.startswith("ingestion_timestamp="):
                add_result(rows, dataset_name, "partition_name", "PASS", partition_name)
            else:
                add_result(rows, dataset_name, "partition_name", "FAIL", partition_name)

            metadata_row_count = int(metadata.get("row_count", -1))
            actual_row_count = count_csv_rows(raw_file)
            if metadata_row_count == actual_row_count and actual_row_count > 0:
                add_result(rows, dataset_name, "row_count_matches_metadata", "PASS", str(actual_row_count))
            else:
                add_result(
                    rows,
                    dataset_name,
                    "row_count_matches_metadata",
                    "FAIL",
                    f"metadata={metadata_row_count}, actual={actual_row_count}",
                )

            metadata_checksum = str(metadata.get("checksum_sha256", ""))
            actual_checksum = file_sha256(raw_file)
            if metadata_checksum == actual_checksum:
                add_result(rows, dataset_name, "checksum_matches_metadata", "PASS", actual_checksum)
            else:
                add_result(
                    rows,
                    dataset_name,
                    "checksum_matches_metadata",
                    "FAIL",
                    f"metadata={metadata_checksum}, actual={actual_checksum}",
                )

            expected_source = "retailrocket_batch"
            if metadata.get("source_system") == expected_source:
                add_result(rows, dataset_name, "source_system", "PASS", expected_source)
            else:
                add_result(rows, dataset_name, "source_system", "FAIL", str(metadata.get("source_system")))
        except Exception as exc:
            add_result(rows, dataset_name, "raw_batch_validation", "FAIL", str(exc))


def validate_api_landing(rows: list[dict[str, Any]]) -> None:
    dataset_name = "raw_api_item_properties_delta"

    try:
        ingestion_dirs = all_ingestion_dirs(RAW_API_DIR, API_TYPE)
        total_records = 0

        add_result(
            rows,
            dataset_name,
            "partition_count_positive",
            "PASS",
            f"Found {len(ingestion_dirs)} API delta partitions",
        )

        for ingestion_dir in ingestion_dirs:
            partition_dataset_name = f"raw_api_item_properties_delta:{ingestion_dir.name}"
            metadata = read_metadata(ingestion_dir)

            expected_files = [
                ingestion_dir / "raw_response.json",
                ingestion_dir / "item_properties_delta.json",
                ingestion_dir / "item_properties_delta.csv",
                ingestion_dir / "item_properties_delta.parquet",
            ]

            for path in expected_files:
                add_result(
                    rows,
                    partition_dataset_name,
                    f"{path.name}_exists",
                    "PASS" if path.exists() else "FAIL",
                    display_path(path),
                )

            raw_response_path = ingestion_dir / "raw_response.json"
            payload = json.loads(raw_response_path.read_text(encoding="utf-8"))
            if {"records", "next_cursor", "has_more"}.issubset(payload.keys()):
                add_result(rows, partition_dataset_name, "raw_response_contract", "PASS", "records, next_cursor, has_more present")
            else:
                add_result(rows, partition_dataset_name, "raw_response_contract", "FAIL", str(sorted(payload.keys())))

            records = payload.get("records", [])
            metadata_row_count = int(metadata.get("row_count", -1))
            if isinstance(records, list) and len(records) == metadata_row_count:
                add_result(rows, partition_dataset_name, "row_count_matches_metadata", "PASS", str(len(records)))
            else:
                add_result(
                    rows,
                    partition_dataset_name,
                    "row_count_matches_metadata",
                    "FAIL",
                    f"records={len(records) if isinstance(records, list) else 'not_list'}, metadata={metadata_row_count}",
                )

            parquet_df = pd.read_parquet(ingestion_dir / "item_properties_delta.parquet")
            total_records += len(parquet_df)
            required_columns = {"timestamp", "itemid", "property", "value", "source_system", "ingestion_timestamp"}
            missing_columns = sorted(required_columns - set(parquet_df.columns))
            if not missing_columns:
                add_result(rows, partition_dataset_name, "normalized_parquet_schema", "PASS", str(sorted(required_columns)))
            else:
                add_result(rows, partition_dataset_name, "normalized_parquet_schema", "FAIL", str(missing_columns))

            if len(parquet_df) == metadata_row_count and len(parquet_df) > 0:
                add_result(rows, partition_dataset_name, "normalized_parquet_row_count", "PASS", str(len(parquet_df)))
            else:
                add_result(
                    rows,
                    partition_dataset_name,
                    "normalized_parquet_row_count",
                    "FAIL",
                    f"parquet={len(parquet_df)}, metadata={metadata_row_count}",
                )

        add_result(
            rows,
            dataset_name,
            "total_partition_rows_positive",
            "PASS" if total_records > 0 else "FAIL",
            f"Total rows across API partitions: {total_records}",
        )

        state_file = RAW_API_DIR / "_state/item_properties_delta_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            add_result(rows, dataset_name, "state_file_exists", "PASS", display_path(state_file))
            add_result(rows, dataset_name, "state_cursor", "PASS", str(state.get("cursor")))
        else:
            add_result(rows, dataset_name, "state_file_exists", "FAIL", display_path(state_file))
    except Exception as exc:
        add_result(rows, dataset_name, "raw_api_validation", "FAIL", str(exc))


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    validate_batch_landing(rows)
    validate_api_landing(rows)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    failed_count = int((report_df["status"] == "FAIL").sum())
    warning_count = int((report_df["status"] == "WARN").sum()) if "WARN" in set(report_df["status"]) else 0

    print(f"Retailrocket raw validation report saved: {display_path(REPORT_PATH)}")
    print(f"Total checks: {len(report_df)}")
    print(f"Failed checks: {failed_count}")
    print(f"Warnings: {warning_count}")

    if failed_count > 0:
        print(report_df[report_df["status"] == "FAIL"].to_string(index=False))
        raise RuntimeError(f"Retailrocket raw validation failed with {failed_count} failed checks.")

    print("\nNo failed Retailrocket raw validation checks.")


if __name__ == "__main__":
    main()
