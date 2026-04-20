import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/source=dummyjson")
REPORT_PATH = Path("reports/dummyjson_raw_summary.csv")


def infer_record_count(data: dict) -> int:
    """
    This function finds the list inside the JSON and returns its length.
    """
    for value in data.values():
        if isinstance(value, list):
            return len(value)

    return 0


def find_list_key(data: dict) -> str | None:
    """
    Finds the key whose value is a list.
    """
    for key, value in data.items():
        if isinstance(value, list):
            return key

    return None


def inspect_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    top_level_keys = list(data.keys())
    list_key = find_list_key(data)
    record_count = infer_record_count(data)

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "top_level_keys": ", ".join(top_level_keys),
        "list_key": list_key,
        "record_count": record_count,
    }


def main() -> None:
    json_files = sorted(RAW_DIR.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found under {RAW_DIR}")
        return

    summaries = []

    for file_path in json_files:
        summary = inspect_file(file_path)
        summaries.append(summary)

        print("\nFile:", summary["file_path"])
        print("Top-level keys:", summary["top_level_keys"])
        print("List key:", summary["list_key"])
        print("Record count:", summary["record_count"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(REPORT_PATH, index=False)

    print(f"\nSaved summary report: {REPORT_PATH}")


if __name__ == "__main__":
    main()