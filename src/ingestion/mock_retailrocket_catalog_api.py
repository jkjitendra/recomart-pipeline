from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOCK_SOURCE_PATH = PROJECT_ROOT / "data/external/retailrocket/item_properties_part2.csv"

app = FastAPI(
    title="RecoMart Retailrocket Catalog Metadata Mock API",
    description=(
        "Mock/demo API that serves Retailrocket item property records from "
        "data/external/retailrocket/item_properties_part2.csv. It implements "
        "GET /api/item-properties?limit=10&cursor=<cursor> for local ingestion tests. "
        "The count query parameter is accepted as an alias for limit so client examples "
        "match the teammate-hosted API contract."
    ),
    version="1.0.0",
)


def parse_cursor(cursor: str | None) -> int:
    if cursor in (None, ""):
        return 0

    try:
        return max(int(cursor), 0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="cursor must be an integer offset") from exc


def coerce_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "timestamp": int(row["timestamp"]),
        "itemid": int(row["itemid"]),
        "property": str(row["property"]),
        "value": str(row["value"]),
    }


def read_page(limit: int, cursor: str | None) -> dict[str, Any]:
    if not MOCK_SOURCE_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Mock source file not found: {MOCK_SOURCE_PATH}")

    offset = parse_cursor(cursor)
    records: list[dict[str, Any]] = []
    has_more = False

    with MOCK_SOURCE_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader):
            if index < offset:
                continue

            if len(records) < limit:
                records.append(coerce_record(row))
                continue

            has_more = True
            break

    return {
        "records": records,
        "next_cursor": str(offset + len(records)),
        "has_more": has_more,
    }


@app.get("/api/item-properties")
def get_item_properties(
    limit: int = Query(default=10, ge=1, le=1000),
    count: int | None = Query(default=None, ge=1, le=1000),
    cursor: str | None = Query(default=None),
) -> dict[str, Any]:
    effective_limit = count if count is not None else limit
    return read_page(limit=effective_limit, cursor=cursor)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ingestion.mock_retailrocket_catalog_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
