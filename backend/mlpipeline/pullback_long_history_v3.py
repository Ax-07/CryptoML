"""Strict read-only 21-month P4 development loader.

Never opens ADA or 2026-07 and later candle partitions, even if these
exist in local manifests. Does not alter the frozen seven-month P4 V1/V2
experiments. Historical development only, not independent OOS.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import REQUIRED
from mlpipeline.trend_pullback import _validate_closed_m5

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
FIRST_MONTH = "2024-10"
LAST_MONTH = "2026-06"
EXPECTED_MONTHS = tuple(
    str(period) for period in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
)
CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
if len(EXPECTED_MONTHS) != 21:
    raise AssertionError("Unexpected long-history calendar geometry")


def load_long_development_history(
    store: CandleStore,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return checked OHLC M5 bars for exactly 4 x 21 approved month partitions.

    Only 2024-10 through 2026-06 inclusive. Verify complete month manifests,
    P0 quality, SHA-256, row counts, candle finality, UTC and OHLC geometry.
    Missing M5 bars are never synthesized: V1 feature detector resets at gaps.
    The final June bar closing at the exclusive July cutoff is dropped *after*
    verifying the entire June partition's original checksum.
    """
    all_manifests = store.list_manifests()  # Metadata only; does not read protected Parquets.
    outputs: dict[str, pd.DataFrame] = {}
    selected_manifests: list[dict] = []

    for symbol in SYMBOLS:
        approved = sorted(
            (m for m in all_manifests
             if m.get("symbol") == symbol and m.get("period") in EXPECTED_MONTHS),
            key=lambda m: m["period"],
        )
        months = tuple(m["period"] for m in approved)
        if months != EXPECTED_MONTHS:
            raise ValueError(
                f"{symbol}: missing/duplicate approved development months; "
                f"expected {len(EXPECTED_MONTHS)}, observed {len(approved)}"
            )

        frames = []
        for manifest in approved:
            month = manifest["period"]
            sha = manifest.get("parquet_sha256")
            if (manifest.get("quality_status") != "OK"
                    or not isinstance(sha, str)
                    or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                    or not isinstance(manifest.get("row_count"), int)
                    or manifest["row_count"] < 1):
                raise ValueError(f"{symbol} {month}: unsafe P0 manifest")
            year, mon = map(int, month.split("-"))
            path = store.partition_dir(symbol, year, mon) / "part-000.parquet"
            if not path.is_file():
                raise ValueError(f"Missing expected P0 file: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != sha:
                raise ValueError(f"P0 SHA-256 checksum mismatch: {path}")
            frame = pd.read_parquet(path, columns=list(REQUIRED))
            if len(frame) != manifest["row_count"]:
                raise ValueError(f"{symbol} {month}: P0 row-count mismatch")
            for col in ("bar_open_at", "bar_close_at"):
                frame[col] = pd.to_datetime(frame[col], utc=True)
            if frame.empty or not frame.bar_open_at.dt.strftime("%Y-%m").eq(month).all():
                raise ValueError(f"{symbol} {month}: candles outside month")
            if (frame.bar_close_at >= CUTOFF).any():
                crossing = frame.bar_close_at >= CUTOFF
                legitimate = (
                    month == LAST_MONTH
                    and int(crossing.sum()) == 1
                    and frame.loc[crossing, "bar_open_at"].eq(
                        CUTOFF - pd.Timedelta(minutes=5)
                    ).all()
                    and frame.loc[crossing, "bar_close_at"].eq(CUTOFF).all()
                )
                if not legitimate:
                    raise ValueError(f"{symbol} {month}: reserved-period crossover")
                frame = frame.loc[~crossing].copy()
            if frame.empty:
                raise ValueError(f"{symbol} {month}: no eligible closed bars")
            frame["symbol"] = symbol
            # Both month-level and joined geometry checks catch invalid OHLC
            # and overlapping/duplicate bars without any imputation.
            frames.append(_validate_closed_m5(frame))
            selected_manifests.append(manifest)

        joined = _validate_closed_m5(pd.concat(frames, ignore_index=True))
        if (not joined.bar_close_at.ge(
                pd.Timestamp("2024-10-01T00:00:00Z")
            ).all() or not joined.bar_close_at.lt(CUTOFF).all()):
            raise AssertionError(f"{symbol}: out-of-scope bar after validation")
        outputs[symbol] = joined

    if len(selected_manifests) != 84:
        raise AssertionError("Expected exactly 84 selected monthly P0 manifests")
    return outputs, selected_manifests
