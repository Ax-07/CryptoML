"""Actual PyArrow smoke test (skips only if pyarrow is absent from dev environment)."""
from datetime import datetime, timezone

import pytest

from ingestion.schema import normalize_klines
from ingestion.storage import CandleStore


def test_real_pyarrow_round_trip(tmp_path, sample_row):
    pytest.importorskip("pyarrow")
    df = normalize_klines(
        [sample_row(1785542400000)], symbol="ETHUSDT", source="rest",
        as_of=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    store = CandleStore(tmp_path)
    store.write_batch(df)
    bars = store.read_candles("ETHUSDT", datetime(2026, 8, 1, tzinfo=timezone.utc),
                              datetime(2026, 8, 2, tzinfo=timezone.utc))
    assert len(bars) == 1
    assert bars.iloc[0]["bar_open_at"].tzinfo is not None
