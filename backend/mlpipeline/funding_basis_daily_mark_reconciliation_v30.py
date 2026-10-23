"""V30: read-only SHA checked monthly-vs-daily markPriceKlines reconciliation.

Never merges, writes, imputes or promotes a market partition. No trading.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from mlpipeline.funding_basis_archive_integrity_v19 import (
    BASE, PATHS, ZIP_LIMIT, fetch, checksum,
)
from mlpipeline.funding_archive_schema_v20 import verified_csv
from mlpipeline.funding_basis_verified_partition_v23 import epoch, parse_kline

PROTOCOL = "p4_v30_btc_june2026_daily_mark_reconciliation_v1"
SYMBOL = "BTCUSDT"
MONTH = "2026-06"
DAY = "2026-06-29"
BEGIN = pd.Timestamp("2026-06-01T00:00:00Z")
END = pd.Timestamp("2026-07-01T00:00:00Z")
DAY_BEGIN = pd.Timestamp("2026-06-29T00:00:00Z")
DAY_END = DAY_BEGIN + pd.Timedelta(days=1)
M5 = pd.Timedelta(minutes=5)
DAILY_PATH = ("futures/um/daily/markPriceKlines/BTCUSDT/5m/"
              "BTCUSDT-5m-2026-06-29.zip")


def verified_archive(url: str) -> tuple[bytes, dict]:
    """Verify Binance ZIP checksum and safe CSV layout; keep bytes only in memory."""
    blob = fetch(url, ZIP_LIMIT)
    check = fetch(url + ".CHECKSUM", 4096)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != checksum(check):
        raise ValueError("Archive CHECKSUM SHA mismatch: " + url)
    return verified_csv(blob), {
        "url": url,
        "zip_sha256": digest,
        "checksum_sha256": hashlib.sha256(check).hexdigest(),
        "checksum_verified": True,
    }


def read_month_rows(data: bytes) -> tuple[pd.DataFrame, pd.Series, str, bool]:
    raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str, keep_default_na=False,
                      low_memory=False)
    if raw.empty:
        raise ValueError("Empty monthly archive")
    has_header = not raw.iat[0, 0].strip().isdigit()
    rows = raw.iloc[1:].reset_index(drop=True) if has_header else raw
    if rows.empty or rows.shape[1] < 7:
        raise ValueError("Invalid monthly CSV shape")
    times, unit = epoch(rows.iloc[:, 0], BEGIN, END)
    return rows, times, unit, has_header


def reconcile(monthly_csv: bytes, daily_csv: bytes) -> dict:
    rows, month_times, monthly_unit, has_header = read_month_rows(monthly_csv)
    all_expected = pd.date_range(BEGIN, END - M5, freq="5min")
    day_expected = pd.date_range(DAY_BEGIN, DAY_END - M5, freq="5min")
    monthly_index = pd.DatetimeIndex(month_times)
    missing = all_expected.difference(monthly_index)
    off_grid = monthly_index.difference(all_expected)

    detail = {
        "monthly_data_rows": len(rows),
        "monthly_columns": rows.shape[1],
        "monthly_header_detected": has_header,
        "monthly_epoch_unit": monthly_unit,
        "expected_month_m5": len(all_expected),
        "monthly_missing_count": len(missing),
        "missing_first_utc": missing[0].isoformat() if len(missing) else None,
        "missing_last_utc": missing[-1].isoformat() if len(missing) else None,
        "missing_exactly_2026_06_29": missing.equals(day_expected),
        "monthly_off_grid_count": len(off_grid),
    }
    day_times, daily_unit, _ = parse_kline(daily_csv, "mark", DAY_BEGIN, DAY_END)
    daily_index = pd.DatetimeIndex(day_times)
    overlap = monthly_index.intersection(daily_index)
    merged = monthly_index.union(daily_index)
    detail.update({
        "daily_m5_count": len(daily_index),
        "expected_daily_m5": len(day_expected),
        "daily_epoch_unit": daily_unit,
        "daily_first_utc": daily_index[0].isoformat(),
        "daily_last_utc": daily_index[-1].isoformat(),
        "monthly_daily_overlap_count": len(overlap),
        "combined_timeline_complete": merged.equals(all_expected),
    })

    eligible_timeline = (
        missing.equals(day_expected)
        and len(off_grid) == 0
        and len(overlap) == 0
        and len(monthly_index) == len(all_expected) - len(day_expected)
        and monthly_unit == daily_unit
        and merged.equals(all_expected)
    )
    # Check all *present* monthly candles against unmodified V23 parser,
    # one complete day at a time. This does NOT synthesize missing bars.
    failures = []
    if eligible_timeline:
        days = pd.date_range(BEGIN, END - pd.Timedelta(days=1), freq="D")
        for start in days:
            if start == DAY_BEGIN:
                continue
            mask = month_times.dt.normalize().eq(start)
            subset = rows.loc[mask].to_csv(index=False, header=False).encode("utf-8")
            try:
                parse_kline(subset, "mark", start, start + pd.Timedelta(days=1))
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                failures.append({
                    "day": start.date().isoformat(),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:180],
                })
    detail["present_monthly_days_checked"] = 29 if eligible_timeline else 0
    detail["present_monthly_day_failures"] = failures
    detail["timeline_reconciliation_eligible_for_review"] = (
        eligible_timeline and not failures
    )
    detail["status"] = (
        "SOURCE_RECONCILIATION_CANDIDATE_NOT_PUBLISHED"
        if detail["timeline_reconciliation_eligible_for_review"]
        else "NOT_ELIGIBLE_REQUIRE_REVIEW"
    )
    return detail


def run(output: Path, permit_downloads: bool = False) -> dict:
    if not permit_downloads:
        raise ValueError("Explicit --permit-downloads required")
    output = output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require NEW output JSON in existing directory")
    monthly_url = BASE + PATHS["mark"].format(s=SYMBOL, m=MONTH)
    daily_url = BASE + DAILY_PATH
    monthly_csv, monthly_source = verified_archive(monthly_url)
    daily_csv, daily_source = verified_archive(daily_url)
    detail = reconcile(monthly_csv, daily_csv)
    result = {
        "protocol": PROTOCOL,
        "symbol": SYMBOL,
        "month": MONTH,
        "daily_source_date": DAY,
        "monthly_archive": monthly_source,
        "daily_archive": daily_source,
        "reconciliation": detail,
        "no_partition_written": True,
        "no_price_export": True,
        "no_imputation_no_backtest_no_training_no_orders": True,
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="V30 read-only daily mark-price source reconciliation"
    )
    p.add_argument("--permit-downloads", action="store_true")
    p.add_argument("--output-json", type=Path, required=True)
    a = p.parse_args()
    report = run(a.output_json, a.permit_downloads)
    r = report["reconciliation"]
    print("V30 BOTH ARCHIVE CHECKSUMS VERIFIED")
    print("monthly missing", r["monthly_missing_count"],
          "exact June 29 gap", r["missing_exactly_2026_06_29"])
    print("daily M5", r["daily_m5_count"],
          "overlap", r["monthly_daily_overlap_count"],
          "combined complete", r["combined_timeline_complete"])
    print("status", r["status"])
    print("NO DATA PROMOTION / NO PNL / NO ORDERS")


if __name__ == "__main__":
    main()
