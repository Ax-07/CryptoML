"""Read-only P0 endpoints; P1 ML endpoints deliberately not implemented yet."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import get_settings
from ingestion.schema import require_utc, validate_symbol
from ingestion.storage import CandleStore

router = APIRouter(prefix="/api/v1", tags=["CryptoML P0"])


class Candle(BaseModel):
    time: int = Field(description="Opening timestamp, Unix seconds, for Lightweight Charts")
    open: float
    high: float
    low: float
    close: float
    volume: float


def local_store() -> CandleStore:
    return CandleStore(get_settings().data_root)


@router.get("/markets/symbols")
def symbols() -> dict:
    manifests = local_store().list_manifests()
    grouped = defaultdict(list)
    for item in manifests:
        grouped[item["symbol"]].append(item)
    rows = [
        {
            "symbol": symbol, "exchange": "binance", "market": "usdt_perpetual", "timeframe": "5m",
            "first_period": min(m["period"] for m in entries),
            "last_period": max(m["period"] for m in entries),
            "imported_bars": sum(m["row_count"] for m in entries),
            "has_quality_warnings": any(m["quality_status"] != "OK" for m in entries),
        }
        for symbol, entries in sorted(grouped.items())
    ]
    return {"items": rows, "total": len(rows)}


@router.get("/datasets")
def datasets() -> dict:
    """P0 dataset inventory: one dataset per monthly Parquet partition."""
    items = local_store().list_manifests()
    return {"items": items, "total": len(items)}


@router.get("/datasets/{dataset_id}/quality")
def quality(dataset_id: str) -> dict:
    report = local_store().quality_by_dataset_id(dataset_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return report


@router.get("/market/candles", response_model=list[Candle])
def candles(
    symbol: str,
    start: datetime = Query(alias="from"),
    end: datetime = Query(alias="to"),
    timeframe: str = "5m",
    limit: int = Query(500, ge=1, le=1500),
) -> list[Candle]:
    if timeframe != "5m":
        raise HTTPException(status_code=422, detail="Only 5m is supported by P0")
    try:
        symbol = validate_symbol(symbol)
        start = require_utc(start, name="from")
        end = require_utc(end, name="to")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if start >= end:
        raise HTTPException(status_code=422, detail="from must be earlier than to")
    if end - start > timedelta(days=31):
        raise HTTPException(status_code=422, detail="Max query window is 31 days")
    # Historical data are saved only after the exclusive bar_close_at has passed.
    # Keep this guard in case a local parquet is modified outside our pipeline.
    now = datetime.now(timezone.utc)
    result = local_store().read_candles(symbol=symbol, start=start, end=end, limit=limit)
    if result.empty:
        return []
    result = result[result.bar_close_at <= now]
    return [
        Candle(time=int(row.bar_open_at.timestamp()), open=float(row.open),
               high=float(row.high), low=float(row.low), close=float(row.close),
               volume=float(row.volume_base))
        for row in result.itertuples(index=False)
    ]
