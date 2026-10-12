"""Orchestration: download -> validate -> immutable raw -> partitioned Parquet."""
from __future__ import annotations

from datetime import datetime, timezone

from ingestion.schema import normalize_klines, require_utc, validate_symbol
from ingestion.sources import BinanceClient
from ingestion.storage import CandleStore


def ingest_month(
    client: BinanceClient, store: CandleStore, *, symbol: str, month: str, as_of: datetime | None = None
) -> list[dict]:
    symbol = validate_symbol(symbol)
    try:
        parsed = datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("Month must be YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != month:
        raise ValueError("Month must be YYYY-MM")
    as_of = require_utc(as_of or datetime.now(timezone.utc), name="as_of")
    if datetime(parsed.year, parsed.month, 1, tzinfo=timezone.utc) >= as_of:
        raise ValueError("Cannot ingest a future month")
    zip_content, checksum, rows = client.download_month(symbol=symbol, month=month)
    candles = normalize_klines(rows, symbol=symbol, source="archive_monthly", as_of=as_of)
    if candles.empty:
        raise ValueError("Archive does not contain any finalized M5 bars")
    expected_period = candles.bar_open_at.dt.strftime("%Y-%m")
    if (expected_period != month).any():
        raise ValueError("Archive contains candles from unexpected month")
    store.store_archive(symbol, month, zip_content, checksum)
    return store.write_batch(candles)


def ingest_rest(
    client: BinanceClient,
    store: CandleStore,
    *, symbol: str,
    start: datetime,
    end: datetime,
    as_of: datetime | None = None,
) -> list[dict]:
    symbol = validate_symbol(symbol)
    start, end = require_utc(start, name="start"), require_utc(end, name="end")
    as_of = require_utc(as_of or datetime.now(timezone.utc), name="as_of")
    if start >= end:
        raise ValueError("start must be before end")
    if (end - start).days > 31:
        raise ValueError("REST sync is limited to 31 days; use monthly archives for longer history")
    rows = client.fetch_rest(symbol=symbol, start=start, end=end, as_of=as_of)
    candles = normalize_klines(rows, symbol=symbol, source="rest", as_of=as_of)
    candles = candles[(candles.bar_open_at >= start) & (candles.bar_open_at < end)].copy()
    return store.write_batch(candles)
