"""Prepare fresh causal M5 feature snapshots for a frozen CPU model.

Input is a genuine externally collected Parquet of CLOSED Binance-style M5
candles, not labeled training data. This module NEVER fetches from an exchange.
For the final timestamp, recompute features through that timestamp only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.features import (
    FEATURE_COLUMNS, FIVE_MINUTES, build_symbol_features, join_market_context,
)
from mlpipeline.trainer import DEFAULT_SYMBOLS

REQUIRED = {
    "symbol", "bar_open_at", "bar_close_at", "is_final", "ingested_at",
    "open", "high", "low", "close", "volume_base", "volume_quote",
    "trade_count", "taker_buy_volume_base", "taker_buy_volume_quote",
}
NUMERIC = (
    "open", "high", "low", "close", "volume_base", "volume_quote",
    "trade_count", "taker_buy_volume_base", "taker_buy_volume_quote",
)
MIN_CONTIGUOUS_BARS = 240  # ~20 hours, including EMA warm-up.


def _require_utc(s: pd.Series, name: str) -> pd.Series:
    if not isinstance(s.dtype, pd.DatetimeTZDtype) or str(s.dt.tz) != "UTC":
        raise ValueError(f"{name} must be explicitly timezone-aware UTC")
    if s.isna().any():
        raise ValueError(f"{name} contains null timestamps")
    return s


def prepare_fresh_features(
    candles: pd.DataFrame, *, observed_at: pd.Timestamp,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> pd.DataFrame:
    """Recompute the exact v1 feature names on a synchronized fresh M5 close.

    Requiring all symbols synchronously avoids cross-asset stale context.
    No label/target/future return columns may enter this interface.
    """
    if not isinstance(observed_at, pd.Timestamp) or observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    now = observed_at.tz_convert("UTC")
    if not {"BTCUSDT", "ETHUSDT"}.issubset(symbols):
        raise ValueError("BTC and ETH context required")
    if candles.empty:
        raise ValueError("Missing closed candles")
    missing = REQUIRED.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing closed-candle columns: {sorted(missing)}")
    forbidden = {"target", "label_mature_at", "future_return", "actual_return"}
    if forbidden.intersection(candles.columns):
        raise ValueError("Future labels must not enter live inference")
    if candles.symbol.isna().any() or set(candles.symbol) != set(symbols):
        raise ValueError("Exactly the configured symbol universe required")
    if candles.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate symbol/bar timestamp")
    df = candles.copy()
    for col in ("bar_open_at", "bar_close_at", "ingested_at"):
        _require_utc(df[col], col)
    if (df.bar_close_at - df.bar_open_at != FIVE_MINUTES).any():
        raise ValueError("Non-M5 candle duration")
    if ((df.bar_open_at != df.bar_open_at.dt.floor("5min"))
            | (df.bar_close_at != df.bar_close_at.dt.floor("5min"))).any():
        raise ValueError("Unaligned M5 timestamps")
    if df.is_final.isna().any() or not df.is_final.eq(True).all():
        raise ValueError("Unfinalized candles are prohibited")
    if (df.bar_close_at > now).any() or (df.ingested_at > now).any():
        raise ValueError("Future candle or future ingestion timestamp")
    if (df.ingested_at < df.bar_close_at).any():
        raise ValueError("Ingestion timestamps precede completed candles")
    values = df[list(NUMERIC)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Invalid numeric OHLCV")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive prices")
    if (df[["volume_base", "volume_quote", "trade_count",
            "taker_buy_volume_base", "taker_buy_volume_quote"]] < 0).any().any():
        raise ValueError("Negative volumes/trades")
    if ((df.low > df[["open", "close"]].min(axis=1))
            | (df.high < df[["open", "close"]].max(axis=1))
            | (df.low > df.high)).any():
        raise ValueError("Inconsistent OHLC range")

    latest = df.groupby("symbol")["bar_close_at"].max()
    if latest.nunique() != 1:
        raise ValueError("Asynchronous latest closes across symbols")
    timestamp = latest.iloc[0]
    if timestamp != now.floor("5min") or not timestamp <= now < timestamp + FIVE_MINUTES:
        raise ValueError("Most recent closed M5 candle is stale or missing")

    prepared = {}
    for symbol in symbols:
        history = df.loc[df.symbol.eq(symbol)].sort_values("bar_open_at")
        # No repeated or missing bars in the current contiguous feature window.
        if history.bar_open_at.duplicated().any():
            raise ValueError(f"Duplicate bar opens: {symbol}")
        breaks = history.bar_open_at.diff().ne(FIVE_MINUTES)
        segment = breaks.cumsum()
        last_segment = history.loc[segment.eq(segment.iloc[-1])]
        if len(last_segment) < MIN_CONTIGUOUS_BARS:
            raise ValueError(f"Insufficient contiguous closed M5 warmup: {symbol}")
        # Deliberately pass all contiguous history; EMA initialization is causal.
        prepared[symbol] = build_symbol_features(last_segment)

    snapshots = []
    for symbol in symbols:
        row = join_market_context(
            prepared[symbol], prepared["BTCUSDT"], prepared["ETHUSDT"]
        )
        now_row = row.loc[row.bar_close_at.eq(timestamp)]
        if len(now_row) != 1:
            raise ValueError(f"No unique final close features for {symbol}")
        snapshots.append(now_row.loc[:, ["symbol", "bar_close_at", *FEATURE_COLUMNS]])
    features = pd.concat(snapshots, ignore_index=True)
    matrix = features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite closed-bar features")
    features.loc[:, FEATURE_COLUMNS] = features.loc[:, FEATURE_COLUMNS].astype("float32")
    return features[["symbol", "bar_close_at", *FEATURE_COLUMNS]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate externally recorded M5 candles and build causal fresh features"
    )
    parser.add_argument("--closed-candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.closed_candles.suffix.lower() != ".parquet":
        parser.error("Closed M5 candles input must be Parquet")
    if args.output.suffix.lower() != ".parquet":
        parser.error("Features output must be Parquet")
    if args.output.exists():
        parser.error("Feature snapshot already exists: refusing overwrite")
    observed_at = pd.Timestamp(datetime.now(timezone.utc))
    candles = pd.read_parquet(args.closed_candles)
    snapshot = prepare_fresh_features(candles, observed_at=observed_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(args.output, index=False)
    print(f"Feature snapshot: {args.output.resolve()}")
    print("M5 UTC close:", snapshot.bar_close_at.iloc[0].isoformat())
    print("Symbols:", ", ".join(snapshot.symbol.to_list()))
    print("No predictions or exchange orders were created.")


if __name__ == "__main__":
    main()
