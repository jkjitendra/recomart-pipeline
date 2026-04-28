from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query


MOCK_TOTAL_ROWS = 1000
MOCK_BASE_RECORDS = [
    {"timestamp": 1433041200000, "itemid": 183478, "property": "561", "value": "769062"},
    {"timestamp": 1439694000000, "itemid": 132256, "property": "976", "value": "n26.400 1135780"},
    {"timestamp": 1435460400000, "itemid": 420307, "property": "921", "value": "1149317 1257525"},
    {"timestamp": 1431831600000, "itemid": 403324, "property": "917", "value": "1204143"},
    {"timestamp": 1435460400000, "itemid": 230701, "property": "521", "value": "769062"},
    {"timestamp": 1433041200000, "itemid": 286407, "property": "202", "value": "820407"},
    {"timestamp": 1438484400000, "itemid": 256368, "property": "888", "value": "437265 1296497 n24.000"},
    {"timestamp": 1437879600000, "itemid": 307534, "property": "888", "value": "150169 212349 1095303"},
    {"timestamp": 1431226800000, "itemid": 8921, "property": "categoryid", "value": "1188"},
    {"timestamp": 1431831600000, "itemid": 215180, "property": "71", "value": "1096621"},
]

app = FastAPI(
    title="RecoMart Retailrocket Catalog Metadata Mock API",
    description=(
        "Mock/demo API that serves deterministic Retailrocket-like item property records. It implements "
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


def generate_mock_record(index: int) -> dict[str, Any]:
    base_record = MOCK_BASE_RECORDS[index % len(MOCK_BASE_RECORDS)]
    cycle = index // len(MOCK_BASE_RECORDS)

    return {
        "timestamp": int(base_record["timestamp"]),
        "itemid": int(base_record["itemid"]) + cycle,
        "property": str(base_record["property"]),
        "value": str(base_record["value"]),
    }


def read_page(limit: int, cursor: str | None) -> dict[str, Any]:
    offset = parse_cursor(cursor)
    if offset >= MOCK_TOTAL_ROWS:
        raise HTTPException(status_code=416, detail="cursor is beyond the mock data range")

    end_offset = min(offset + limit, MOCK_TOTAL_ROWS)
    records = [generate_mock_record(index) for index in range(offset, end_offset)]

    return {
        "records": records,
        "next_cursor": str(end_offset),
        "has_more": end_offset < MOCK_TOTAL_ROWS,
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
