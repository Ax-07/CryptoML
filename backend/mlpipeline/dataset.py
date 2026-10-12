"""Read existing P0 immutable candle partitions and validate their continuity."""
from __future__ import annotations

import hashlib
from datetime import timezone
from pathlib import Path

import pandas as pd

from ingestion.schema import validate_symbol
from ingestion.storage import CandleStore

REQUIRED = (
    "bar_open_at", "bar_close_at", "open", "high", "low", "close",
    "volume_base", "volume_quote", "trade_count", "taker_buy_volume_base",
    "taker_buy_volume_quote", "is_final",
)


def load_symbol(store: CandleStore, symbol: str, *, months: int | None = None,
                verify_hashes: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Load all/months-most-recent monthly P0 data, no candle fabrication.

    `months` limits *whole partitions*. Data from the previous month is not
    inferred to fill gaps, and no partial current M5 candle is used.
    """
    symbol = validate_symbol(symbol)
    manifests = sorted(
        (item for item in store.list_manifests() if item["symbol"] == symbol),
        key=lambda item: item["period"],
    )
    if months is not None:
        if months < 1:
            raise ValueError("months must be >= 1")
        manifests = manifests[-months:]
    if not manifests:
        raise ValueError(f"No P0 Parquet partitions for {symbol}")
    frames = []
    for manifest in manifests:
        yyyy, mm = map(int, manifest["period"].split("-"))
        path = store.partition_dir(symbol, yyyy, mm) / "part-000.parquet"
        if not path.is_file():
            raise ValueError(f"Manifest without Parquet: {path}")
        if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != manifest["parquet_sha256"]:
            raise ValueError(f"Parquet checksum does not match manifest: {path}")
        if manifest["quality_status"] != "OK":
            raise ValueError(f"P0 partition quality_status != OK: {path}")
        frame = pd.read_parquet(path, columns=list(REQUIRED))
        frame["symbol"] = symbol
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for col in ("bar_open_at", "bar_close_at"):
        data[col] = pd.to_datetime(data[col], utc=True)
    data = data.sort_values("bar_open_at").reset_index(drop=True)
    if data["bar_open_at"].duplicated().any():
        raise ValueError(f"Duplicate M5 timestamps for {symbol}")
    if not data["is_final"].all():
        raise ValueError(f"Unfinalized M5 candle in {symbol}")
    if not ((data["bar_close_at"] - data["bar_open_at"]) == pd.Timedelta(minutes=5)).all():
        raise ValueError(f"Invalid exclusive M5 close in {symbol}")
    if (data["bar_open_at"].diff().dropna() < pd.Timedelta(minutes=5)).any():
        raise ValueError(f"Non-monotonic/overlapping bars in {symbol}")
    return data, manifests


def manifest_fingerprint(manifests: list[dict]) -> str:
    """Deterministic dataset identity based on ingested immutable file hashes."""
    serialized = "\n".join(sorted(f"{m['dataset_id']}:{m['parquet_sha256']}" for m in manifests))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
