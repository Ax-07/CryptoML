"""Canonical Binance USD-M kline normalization (UTC, exclusive bar close)."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

INTERVAL = timedelta(minutes=5)
INTERVAL_MS = 300_000
SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,25}USDT$")
PRICE_COLUMNS = ("open", "high", "low", "close")
VOLUME_COLUMNS = (
    "volume_base", "volume_quote", "taker_buy_volume_base", "taker_buy_volume_quote"
)
VALUE_COLUMNS = [*PRICE_COLUMNS, *VOLUME_COLUMNS, "trade_count"]
CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume_base", "close_time",
    "volume_quote", "trade_count", "taker_buy_volume_base", "taker_buy_volume_quote", "ignore",
]
COLUMNS = [
    "exchange", "market", "symbol", "timeframe", "bar_open_at", "bar_close_at",
    *PRICE_COLUMNS, *VOLUME_COLUMNS, "trade_count", "source", "ingested_at", "is_final",
]


def validate_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not SYMBOL_RE.fullmatch(value):
        raise ValueError("Expected an uppercase USDT pair such as BTCUSDT")
    return value


def require_utc(value: datetime, *, name: str = "timestamp") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """Binance archives can use milliseconds or microseconds; REST uses milliseconds."""
    numeric = pd.to_numeric(series, errors="raise").astype("int64")
    if numeric.empty:
        return pd.to_datetime(numeric, utc=True)
    units = "us" if numeric.abs().max() >= 10**14 else "ms"
    return pd.to_datetime(numeric, unit=units, utc=True, errors="raise")


def normalize_klines(rows: list[list], *, symbol: str, source: str, as_of: datetime) -> pd.DataFrame:
    symbol = validate_symbol(symbol)
    as_of = require_utc(as_of, name="as_of")
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    if any(len(row) < 11 for row in rows):
        raise ValueError("Invalid Binance kline: expected at least 11 fields")
    raw = pd.DataFrame([row[:12] + [None] * max(0, 12 - len(row)) for row in rows], columns=CSV_COLUMNS)
    opened = normalize_timestamp(raw["open_time"])
    if (opened.dt.minute % 5 != 0).any() or (opened.dt.second != 0).any() or (opened.dt.microsecond != 0).any():
        raise ValueError("Found kline whose opening timestamp is not aligned to M5")
    df = pd.DataFrame({
        "exchange": "binance",
        "market": "usdt_perpetual",
        "symbol": symbol,
        "timeframe": "5m",
        "bar_open_at": opened,
        "bar_close_at": opened + INTERVAL,
        "source": source,
        "ingested_at": pd.Timestamp.now(tz="UTC"),
        "is_final": True,
    })
    for field in (*PRICE_COLUMNS, *VOLUME_COLUMNS):
        df[field] = pd.to_numeric(raw[field], errors="raise").astype("float64")
    df["trade_count"] = pd.to_numeric(raw["trade_count"], errors="raise").astype("int64")
    if any(not math.isfinite(float(x)) for field in (*PRICE_COLUMNS, *VOLUME_COLUMNS) for x in df[field]):
        raise ValueError("Non-finite price or volume")
    if (df[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("Price must be strictly positive")
    if (df[list(VOLUME_COLUMNS)] < 0).any().any() or (df["trade_count"] < 0).any():
        raise ValueError("Negative volume or trade count")
    if ((df["low"] > df[["open", "close"]].min(axis=1)) |
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"] > df["high"])).any():
        raise ValueError("Invalid OHLC range")
    # Bar closes when open + 5m <= as_of. No current forming candle is saved.
    df = df.loc[df["bar_close_at"] <= as_of, COLUMNS].sort_values("bar_open_at")
    if df["bar_open_at"].duplicated().any():
        raise ValueError("Duplicate open timestamps in one source batch")
    return df.reset_index(drop=True)
