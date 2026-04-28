from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = PROJECT_ROOT / "data/external/retailrocket"
RAW_ROOT = PROJECT_ROOT / "data/raw/source=retailrocket_batch"
LOG_PATH = PROJECT_ROOT / "logs/retailrocket_batch_ingestion.log"
SUMMARY_PATH = PROJECT_ROOT / "reports/retailrocket_batch_ingestion_summary.csv"

SOURCE_SYSTEM = "retailrocket_batch"

DATASETS = {
    "events": EXTERNAL_DIR / "events.csv",
    "item_properties_part1": EXTERNAL_DIR / "item_properties_part1.csv",
    "category_tree": EXTERNAL_DIR / "category_tree.csv",
}


def configure_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("recomart.retailrocket_batch_ingestion")
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


def ingestion_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def landing_dir(data_type: str, run_timestamp: str) -> Path:
    return RAW_ROOT / f"type={data_type}" / f"ingestion_timestamp={run_timestamp}"


def ensure_sources_exist() -> None:
    missing_files = [str(path) for path in DATASETS.values() if not path.exists()]

    if missing_files:
        raise FileNotFoundError(f"Missing Retailrocket batch source files: {missing_files}")


def write_metadata(
    *,
    output_dir: Path,
    data_type: str,
    source_file: Path,
    raw_file: Path,
    run_timestamp: str,
    row_count: int,
    checksum_sha256: str,
) -> Path:
    metadata = {
        "source_system": SOURCE_SYSTEM,
        "data_type": data_type,
        "source_file": str(source_file.relative_to(PROJECT_ROOT)),
        "raw_file": str(raw_file.relative_to(PROJECT_ROOT)),
        "ingestion_timestamp": run_timestamp,
        "row_count": row_count,
        "checksum_sha256": checksum_sha256,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "landing_path": str(output_dir.relative_to(PROJECT_ROOT)),
    }

    metadata_path = output_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata_path


def ingest_dataset(
    *,
    data_type: str,
    source_file: Path,
    run_timestamp: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_dir = landing_dir(data_type, run_timestamp)

    if output_dir.exists():
        raise FileExistsError(
            f"Raw landing path already exists for this ingestion timestamp: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_file = output_dir / source_file.name

    logger.info("Copying data_type=%s source=%s destination=%s", data_type, source_file, raw_file)
    shutil.copy2(source_file, raw_file)

    row_count = count_csv_rows(raw_file)
    checksum_sha256 = file_sha256(raw_file)

    metadata_path = write_metadata(
        output_dir=output_dir,
        data_type=data_type,
        source_file=source_file,
        raw_file=raw_file,
        run_timestamp=run_timestamp,
        row_count=row_count,
        checksum_sha256=checksum_sha256,
    )

    logger.info(
        "Ingested data_type=%s rows=%s checksum_sha256=%s landing_path=%s",
        data_type,
        row_count,
        checksum_sha256,
        output_dir,
    )

    return {
        "source_system": SOURCE_SYSTEM,
        "data_type": data_type,
        "source_file": str(source_file.relative_to(PROJECT_ROOT)),
        "raw_file": str(raw_file.relative_to(PROJECT_ROOT)),
        "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
        "landing_path": str(output_dir.relative_to(PROJECT_ROOT)),
        "ingestion_timestamp": run_timestamp,
        "row_count": row_count,
        "checksum_sha256": checksum_sha256,
        "file_size_mb": round(raw_file.stat().st_size / (1024 * 1024), 3),
        "status": "PASS",
        "error": "",
    }


def main() -> None:
    logger = configure_logger()
    run_timestamp = ingestion_timestamp()

    logger.info("Starting Retailrocket batch ingestion ingestion_timestamp=%s", run_timestamp)
    ensure_sources_exist()

    rows: list[dict[str, Any]] = []

    for data_type, source_file in DATASETS.items():
        try:
            rows.append(
                ingest_dataset(
                    data_type=data_type,
                    source_file=source_file,
                    run_timestamp=run_timestamp,
                    logger=logger,
                )
            )
        except Exception as exc:
            logger.exception("Failed to ingest data_type=%s source=%s", data_type, source_file)
            rows.append(
                {
                    "source_system": SOURCE_SYSTEM,
                    "data_type": data_type,
                    "source_file": str(source_file.relative_to(PROJECT_ROOT)),
                    "raw_file": "",
                    "metadata_file": "",
                    "landing_path": str(landing_dir(data_type, run_timestamp).relative_to(PROJECT_ROOT)),
                    "ingestion_timestamp": run_timestamp,
                    "row_count": None,
                    "checksum_sha256": "",
                    "file_size_mb": None,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    logger.info("Retailrocket batch ingestion summary saved path=%s", SUMMARY_PATH)

    print("Retailrocket batch ingestion completed.")
    print(f"Ingestion timestamp: {run_timestamp}")
    print(f"Summary saved: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Log saved: {LOG_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(summary_df.to_string(index=False))

    failed_count = int((summary_df["status"] == "FAIL").sum())
    if failed_count > 0:
        raise RuntimeError(f"Retailrocket batch ingestion failed for {failed_count} dataset(s).")


if __name__ == "__main__":
    main()
