"""One-shot public REST observer for genuinely fresh Binance USD-M M5 candles.

No keys, no private endpoints, no orders, no scheduler, no historical backfill.
Reuses the existing CryptoML ingestion client and exact feature calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingestion.schema import normalize_klines
from ingestion.sources import BinanceClient
from mlpipeline.fresh_features import MIN_CONTIGUOUS_BARS, prepare_fresh_features
from mlpipeline.forward_signals import predict_snapshot, write_snapshots_once
from mlpipeline.trainer import DEFAULT_SYMBOLS

FIVE_MINUTES = pd.Timedelta(minutes=5)


def collect_current_closed_m5(
    client: BinanceClient,
    *, started_at: pd.Timestamp,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    lookback_bars: int = 300,
) -> pd.DataFrame:
    """Read only a bounded fresh public window; reject missing newest bars.

    REST endTime is exclusive on candle opening timestamps. We request
    [close - lookback*5m, close), so the last desired OPEN is close - 5m.
    The last candle itself is closed at the chosen UTC 5-minute boundary.
    """
    if not isinstance(started_at, pd.Timestamp) or started_at.tzinfo is None:
        raise ValueError("started_at must include a timezone")
    started_at = started_at.tz_convert("UTC")
    if not 240 <= lookback_bars <= 500:
        raise ValueError("lookback_bars must be in [240, 500]")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Configured symbols must be nonempty and unique")
    if not {"BTCUSDT", "ETHUSDT"}.issubset(symbols):
        raise ValueError("BTC/ETH market context required")
    target_close = started_at.floor("5min")
    start = target_close - lookback_bars * FIVE_MINUTES
    frames = []
    for symbol in symbols:
        rows = client.fetch_rest(
            symbol=symbol, start=start.to_pydatetime(),
            end=target_close.to_pydatetime(),
            as_of=started_at.to_pydatetime(),
        )
        if not rows:
            raise ValueError(f"No public closed M5 candles: {symbol}")
        # normalize_klines only accepts fully closed bars as of started_at;
        # ingestion timestamps are stamped when each public response is parsed.
        current = normalize_klines(
            rows, symbol=symbol, source="rest", as_of=started_at.to_pydatetime()
        )
        current = current.loc[
            current.bar_open_at.ge(start) & current.bar_close_at.le(target_close)
        ]
        if current.empty or current.bar_close_at.max() != target_close:
            raise ValueError(f"Missing latest closed M5 candle: {symbol}")
        if len(current) < MIN_CONTIGUOUS_BARS:
            raise ValueError(f"Insufficient M5 warmup from REST: {symbol}")
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def observe_once(
    client: BinanceClient, *, model_dir: Path | None,
    log_dir: Path | None, check_only: bool = False,
    lookback_bars: int = 300,
) -> dict:
    """Fetch, recompute synchronized features, optionally write future snapshots."""
    if not check_only and (model_dir is None or log_dir is None):
        raise ValueError("model_dir and log_dir required for forward signal recording")
    start = pd.Timestamp(datetime.now(timezone.utc))
    candles = collect_current_closed_m5(
        client, started_at=start, lookback_bars=lookback_bars
    )
    completed = pd.Timestamp(datetime.now(timezone.utc))
    if completed.floor("5min") != start.floor("5min"):
        raise ValueError("Public REST collection crossed M5 boundary; skip this batch")
    features = prepare_fresh_features(candles, observed_at=completed)
    summary = {
        "status": "PUBLIC_M5_CHECK_ONLY" if check_only else "FORWARD_SIGNALS_LOGGED",
        "target_close_utc": features.bar_close_at.iloc[0].isoformat(),
        "symbols": features.symbol.tolist(),
        "candles": len(candles),
        "feature_rows": len(features),
    }
    if check_only:
        return summary
    # The prediction time is captured by the logger, not supplied from disk.
    rows = predict_snapshot(
        model_dir, features, recorded_at=pd.Timestamp(datetime.now(timezone.utc))
    )
    paths = write_snapshots_once(rows, log_dir)
    summary["model_id"] = rows[0]["model_id"]
    summary["snapshot_files"] = [str(path) for path in paths]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Binance Futures M5 observer; optional immutable predictions"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--lookback-bars", type=int, default=300)
    args = parser.parse_args()
    if not args.check_only and (not args.model_dir or not args.log_dir):
        parser.error("--model-dir and --log-dir required unless --check-only")
    with BinanceClient() as client:
        result = observe_once(
            client, model_dir=args.model_dir, log_dir=args.log_dir,
            check_only=args.check_only, lookback_bars=args.lookback_bars,
        )
    print("Status:", result["status"])
    print("Close UTC:", result["target_close_utc"])
    print("Symbols:", ", ".join(result["symbols"]))
    print("Closed candles:", result["candles"])
    print("Feature rows:", result["feature_rows"])
    if "model_id" in result:
        print("Frozen model:", result["model_id"])
        print("Snapshot log:", args.log_dir.resolve())
    print("No private API, exchange order, or trade execution was used.")


if __name__ == "__main__":
    main()
