"""Offline synthetic tests of P4 V3 long-history loader and reserve fences."""
import hashlib

import pandas as pd
import pytest

from ingestion.storage import CandleStore
from mlpipeline.pullback_long_history_v3 import (
    EXPECTED_MONTHS, SYMBOLS, CUTOFF, load_long_development_history,
)


def populate(tmp_path):
    store = CandleStore(tmp_path)
    manifests = []
    for symbol in SYMBOLS:
        for month in EXPECTED_MONTHS:
            year, mon = map(int, month.split("-"))
            start = pd.Timestamp(f"{month}-01T00:00:00Z")
            # This is a synthetic minimal monthly partition, not actual data.
            opens = pd.date_range(start, periods=45, freq="5min")
            if month == "2026-06":
                opens = opens.append(pd.DatetimeIndex([CUTOFF - pd.Timedelta(minutes=5)]))
            price = 100.0
            df = pd.DataFrame({
                "bar_open_at": opens,
                "bar_close_at": opens + pd.Timedelta(minutes=5),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.25,
                "volume_base": 1.0, "volume_quote": 100.0,
                "trade_count": 10, "taker_buy_volume_base": 0.5,
                "taker_buy_volume_quote": 50.0, "is_final": True,
            })
            directory = store.partition_dir(symbol, year, mon)
            directory.mkdir(parents=True, exist_ok=True)
            file = directory / "part-000.parquet"
            df.to_parquet(file, index=False)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            manifest = {"symbol": symbol, "period": month, "row_count": len(df),
                        "quality_status": "OK", "parquet_sha256": digest,
                        "dataset_id": f"{symbol}-{month}"}
            import json
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            manifests.append(manifest)
    return store


def test_reads_only_approved_84_partitions(tmp_path):
    store = populate(tmp_path)
    data, manifests = load_long_development_history(store)
    assert set(data) == set(SYMBOLS)
    assert len(manifests) == 84
    assert all(frame.bar_close_at.lt(CUTOFF).all() for frame in data.values())
    assert all(len(frame) == 21 * 45 for frame in data.values())
    assert data["BTCUSDT"].bar_open_at.iloc[0].year == 2024


def test_missing_month_fails_closed(tmp_path):
    store = populate(tmp_path)
    directory = store.partition_dir("BTCUSDT", 2024, 10)
    (directory / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing/duplicate"):
        load_long_development_history(store)


def test_detects_modified_parquet_hash(tmp_path):
    store = populate(tmp_path)
    directory = store.partition_dir("BTCUSDT", 2024, 10)
    with (directory / "part-000.parquet").open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_long_development_history(store)


def test_rejects_nonfinal_bars(tmp_path):
    store = populate(tmp_path)
    directory = store.partition_dir("BTCUSDT", 2024, 10)
    path = directory / "part-000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "is_final"] = False
    frame.to_parquet(path, index=False)
    import json
    mfile = directory / "manifest.json"
    m = json.loads(mfile.read_text(encoding="utf-8"))
    m["parquet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    mfile.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="closed five-minute"):
        load_long_development_history(store)
