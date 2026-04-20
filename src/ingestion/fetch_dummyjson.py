import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://dummyjson.com"
RAW_DIR = Path("data/raw/source=dummyjson")
LOG_FILE = Path("logs/ingestion_dummyjson.log")


LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def fetch_endpoint(endpoint: str, limit: int = 100, retries: int = 3) -> dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}?limit={limit}"

    for attempt in range(1, retries + 1):
        try:
            logging.info("Fetching endpoint=%s attempt=%s url=%s", endpoint, attempt, url)

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            logging.info("Successfully fetched endpoint=%s", endpoint)

            return data

        except requests.RequestException as exc:
            logging.warning(
                "Failed endpoint=%s attempt=%s error=%s",
                endpoint,
                attempt,
                exc,
            )

            if attempt == retries:
                logging.exception("All retries failed for endpoint=%s", endpoint)
                raise

    raise RuntimeError(f"Unexpected failure while fetching {endpoint}")


def save_raw_json(data: dict[str, Any], endpoint: str) -> Path:
    ingest_date = datetime.now().strftime("%Y-%m-%d")
    ingest_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = RAW_DIR / f"type={endpoint}" / f"ingest_date={ingest_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{endpoint}_{ingest_ts}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    logging.info("Saved raw file=%s", output_path)

    return output_path


def main() -> None:
    endpoints = ["products", "users", "carts"]

    for endpoint in endpoints:
        try:
            data = fetch_endpoint(endpoint)
            output_path = save_raw_json(data, endpoint)
            print(f"Saved {endpoint}: {output_path}")

        except Exception as exc:
            print(f"Failed {endpoint}: {exc}")
            logging.exception("Failed endpoint=%s", endpoint)


if __name__ == "__main__":
    main()