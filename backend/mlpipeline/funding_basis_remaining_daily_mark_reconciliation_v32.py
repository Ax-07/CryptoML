"""V32: read-only reconciliation for ETH/SOL/XRP June 2026 mark M5.

This is a source audit only. It NEVER publishes/fills a historical partition,
changes the validated 80, or computes PnL, training or orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlpipeline.funding_basis_archive_integrity_v19 import BASE, PATHS
from mlpipeline.funding_basis_daily_mark_reconciliation_v30 import (
    DAILY_PATH, MONTH, DAY, reconcile, verified_archive,
)

PROTOCOL = "p4_v32_remaining_june2026_daily_mark_reconciliation_v1"
SYMBOLS = ("ETHUSDT", "SOLUSDT", "XRPUSDT")
# Immutable inputs observed by checksum-verified V31 preflight. If they change,
# stop and investigate rather than silently accepting a new monthly revision.
EXPECTED_MONTHLY_SHA256 = {
    "ETHUSDT": "de38736aea72c09d2802071b7c49bb108557fc5a5c81022bf8d706fd7e42430d",
    "SOLUSDT": "0af68742b32f759dd96a0b93e945184ad85a61a630e09a16e9072174a46cfa55",
    "XRPUSDT": "47c62563101585f9749763bca4b9b1813ac4ddfdcd1232dd074e54baa8244fcd",
}


def urls(symbol: str) -> tuple[str, str]:
    if symbol not in SYMBOLS:
        raise ValueError("Symbol outside frozen remaining June universe")
    monthly = BASE + PATHS["mark"].format(s=symbol, m=MONTH)
    daily = BASE + DAILY_PATH.replace("BTCUSDT", symbol)
    return monthly, daily


def run(output: Path, permit_downloads: bool = False) -> dict:
    if not permit_downloads:
        raise ValueError("Explicit --permit-downloads required")
    output = output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require a NEW JSON report in an existing directory")
    reports = []
    for symbol in SYMBOLS:
        month_url, day_url = urls(symbol)
        monthly, month_source = verified_archive(month_url)
        if month_source["zip_sha256"] != EXPECTED_MONTHLY_SHA256[symbol]:
            raise ValueError("Monthly ZIP changed since V31: " + symbol)
        daily, day_source = verified_archive(day_url)
        result = reconcile(monthly, daily)
        reports.append({
            "symbol": symbol, "month": MONTH, "day": DAY,
            "monthly_archive": month_source,
            "daily_archive": day_source,
            "reconciliation": result,
        })
    all_eligible = all(
        row["reconciliation"]["timeline_reconciliation_eligible_for_review"]
        for row in reports
    )
    out = {
        "protocol": PROTOCOL, "reports": reports,
        "all_source_candidates_eligible": all_eligible,
        "status": ("ALL_SOURCE_CANDIDATES_FOR_REVIEW_NOT_PUBLISHED"
                   if all_eligible else "SOURCE_REVIEW_REQUIRED"),
        "no_partition_written": True,
        "no_price_export": True,
        "no_imputation_no_training_no_pnl_no_orders": True,
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="V32 strict read-only daily mark-source reconciliation")
    p.add_argument("--permit-downloads", action="store_true")
    p.add_argument("--output-json", required=True, type=Path)
    a = p.parse_args()
    output = run(a.output_json, a.permit_downloads)
    for r in output["reports"]:
        v = r["reconciliation"]
        print(r["symbol"], "monthly missing", v["monthly_missing_count"],
              "day exactly 29th", v["missing_exactly_2026_06_29"],
              "daily", v["daily_m5_count"],
              "overlap", v["monthly_daily_overlap_count"],
              "full", v["combined_timeline_complete"],
              "eligible", v["timeline_reconciliation_eligible_for_review"],
              flush=True)
    print(output["status"], flush=True)
    print("NO PARTITION WRITE / NO PNL / NO ORDERS", flush=True)


if __name__ == "__main__":
    main()
