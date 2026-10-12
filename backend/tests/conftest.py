"""Offline tests remain runnable without downloading Binance market data."""
from datetime import datetime, timezone

import pytest


def row(open_ms: int, close: str = "102.0", volume: str = "10.0") -> list:
    return [open_ms, "100", "103", "99", close, volume,
            open_ms + 300_000 - 1, "1010", 22, "5", "510", "0"]


@pytest.fixture
def sample_row():
    return row


@pytest.fixture
def mock_parquet(monkeypatch):
    """Local serializer shim to exercise logic when Arrow is not available.

    Real pyarrow integration is independently covered in test_real_parquet.py.
    """
    import pandas as pd

    def write(df, path, *args, **kwargs):
        return df.to_pickle(path)

    def read(path, *args, **kwargs):
        return pd.read_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write)
    monkeypatch.setattr(pd, "read_parquet", read)
