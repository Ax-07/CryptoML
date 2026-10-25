"""P4 V34: offline provenance-pinned V21/V23/V33 development reader and audit.

Reads ONLY four approved symbols and 2024-10..2026-06, never OOS months.
No network, raw price export, feature generation, training, backtest, PnL or orders.
Uses the V33 provenance freeze and fully replays existing native-source audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_partition_registry_v24 import registry as legacy_registry
from mlpipeline.funding_basis_controlled_batches_v27 import snapshot
from mlpipeline.funding_basis_dual_native_ingestion_v33 import (
    TARGETS, MONTH as LAST_MONTH, PROTOCOL as V33_PROTOCOL,
    AUDIT_PROTOCOL as V33_AUDIT_PROTOCOL,
    checked_roots, audit_records, audit_counts, sha_file,
)
from mlpipeline.pullback_long_history_v3 import (
    SYMBOLS, EXPECTED_MONTHS, FIRST_MONTH, CUTOFF,
)

PROTOCOL = "p4_v34_unified_frozen_development_reader_audit_v1"
FREEZE_PROTOCOL = "p4_v33_final_provenance_freeze_v1"
PLAN_PROTOCOL = "p4_v33_approved_dual_source_plan_v1"
APPROVED_FREEZE_SHA256 = (
    "36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71"
)
APPROVED_PLAN_SHA256 = (
    "4cd4305d00f58095a0dfdc834a88ad2cd60ca14aef5fdb3484f0f9e3dbf56e6e"
)
APPROVED_PLAN_NAME = (
    "p4_v33_approved_dual_native_plan_20261024_141102_5549721.json"
)
APPROVED_V30_NAME = (
    "p4_v30_btc_2026_06_daily_mark_reconciliation_20261024_102253_8061889.json"
)
APPROVED_V32_NAME = (
    "p4_v32_june2026_mark_daily_20261024_115506_3288142.json"
)
APPROVED_AUDIT_JSON_NAME = (
    "p4_v33_final_audit_20261024_142724_9280494.json"
)
APPROVED_AUDIT_CSV_NAME = (
    "p4_v33_final_audit_20261024_142724_9280494.csv"
)
FROZEN_COUNTS = {
    "VERIFIED_V21": 1, "VERIFIED_V23": 79,
    "VERIFIED_V33": 4, "MISSING": 0, "FAILED_INTEGRITY": 0,
}
MARKET_FIELDS = (
    "bar_open_at", "bar_close_at",
    *(f"spot_{v}" for v in ("open", "high", "low", "close", "volume")),
    *(f"perp_{v}" for v in ("open", "high", "low", "close", "volume")),
    *(f"mark_{v}" for v in ("open", "high", "low", "close")),
)
FUNDING_FIELDS = (
    "funding_at", "funding_rate_fraction", "funding_interval_hours",
)
M5 = pd.Timedelta(minutes=5)
BEGIN = pd.Timestamp(FIRST_MONTH + "-01T00:00:00Z")
KIND_FILES = (
    "manifest.json", "markets_5m.parquet", "funding_events.parquet",
    "mark_reconstructed.csv",
    *(f"raw/{kind}.zip" for kind in ("spot", "perp", "mark_monthly", "mark_daily", "funding")),
    *(f"raw/{kind}.zip.CHECKSUM" for kind in ("spot", "perp", "mark_monthly", "mark_daily", "funding")),
)


def _approved_key(symbol: str, month: str) -> None:
    if symbol not in SYMBOLS or month not in EXPECTED_MONTHS:
        raise ValueError("Outside approved historical development universe / holdout refused")


def _month_times(month: str):
    begin = pd.Timestamp(month + "-01T00:00:00Z")
    end = begin + pd.offsets.MonthBegin(1)
    return begin, end


def _check_hash(name: str, actual: str) -> None:
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise ValueError("Malformed provenance SHA-256: " + name)


def _one_named_file(evidence, name: str, base: Path) -> Path:
    matches = [Path(item["path"]).resolve() for item in evidence
               if Path(item["path"]).name == name]
    if len(matches) != 1 or matches[0].parent != base:
        raise ValueError("Missing/duplicate or unexpected evidence path: " + name)
    return matches[0]


def _load_and_check_freeze(freeze_path: Path, approved_sha: str):
    freeze_path = freeze_path.resolve()
    if approved_sha != APPROVED_FREEZE_SHA256:
        raise ValueError("Frozen V33 SHA does not match preregistered V34 baseline")
    if not freeze_path.is_file() or sha_file(freeze_path) != approved_sha:
        raise ValueError("V33 provenance freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_bytes())
    if (freeze.get("protocol") != FREEZE_PROTOCOL
            or freeze.get("approved_plan_sha256") != APPROVED_PLAN_SHA256
            or freeze.get("verified_partitions") != 84
            or freeze.get("missing_partitions") != 0
            or freeze.get("failed_integrity") != 0
            or freeze.get("no_data_modification") is not True):
        raise ValueError("Invalid frozen V33 provenance identity")
    evidence = freeze.get("evidence")
    partitions = freeze.get("v33_partition_files")
    if not isinstance(evidence, list) or len(evidence) != 7:
        raise ValueError("Expected exactly seven V33 proof files")
    if not isinstance(partitions, list) or len(partitions) != 56:
        raise ValueError("Expected exactly 56 original V33 partition files")
    base = freeze_path.parent
    known_names = (
        APPROVED_PLAN_NAME, APPROVED_V30_NAME, APPROVED_V32_NAME,
        APPROVED_AUDIT_JSON_NAME, APPROVED_AUDIT_CSV_NAME,
    )
    fixed = {name: _one_named_file(evidence, name, base) for name in known_names}
    rest = [Path(item["path"]).resolve() for item in evidence
            if Path(item["path"]).name not in known_names]
    if (len(rest) != 2 or any(p.parent != base for p in rest)
            or not any(re.fullmatch(r"p4_v33_batch_summary_\d{8}_\d{6}_\d{7}\.json",
                                    p.name) for p in rest)
            or not any(re.fullmatch(r"p4_v33_batch_journal_\d{8}_\d{6}_\d{7}\.jsonl",
                                    p.name) for p in rest)):
        raise ValueError("Unexpected V33 batch proof files")
    summary_file = next(p for p in rest if "summary" in p.name)
    journal_file = next(p for p in rest if "journal" in p.name)
    if (summary_file.name.replace("summary", "journal").replace(".json", ".jsonl")
            != journal_file.name):
        raise ValueError("V33 journal/summary runs differ")
    known_paths = set(fixed.values()) | set(rest)
    if len(known_paths) != 7:
        raise ValueError("Duplicate V33 evidence paths")
    for item in evidence:
        if set(item) != {"path", "sha256"}:
            raise ValueError("Unexpected evidence fields")
        path = Path(item["path"]).resolve()
        _check_hash("evidence", item["sha256"])
        if path not in known_paths or not path.is_file() or sha_file(path) != item["sha256"]:
            raise ValueError("Frozen evidence missing or SHA mismatch: " + str(path))
    plan_path = fixed[APPROVED_PLAN_NAME]
    if sha_file(plan_path) != APPROVED_PLAN_SHA256:
        raise ValueError("Frozen V33 plan SHA mismatch")
    plan = json.loads(plan_path.read_bytes())
    if (plan.get("protocol") != PLAN_PROTOCOL
            or plan.get("initial_verified") != 80
            or plan.get("initial_missing") != 4
            or plan.get("max_new") != 4
            or plan.get("targets") != [
                {"symbol": symbol, "month": LAST_MONTH} for symbol in TARGETS
            ]):
        raise ValueError("Unexpected V33 source-plan targets")
    sources = plan["sources"]
    if (Path(sources["v30_report"]).resolve() != fixed[APPROVED_V30_NAME]
            or Path(sources["v32_report"]).resolve() != fixed[APPROVED_V32_NAME]):
        raise ValueError("Original source reports differ from plan")
    roots = checked_roots(*(Path(sources[k]) for k in (
        "data_root", "v21_directory", "v23_root", "v33_root",
        "coverage_json", "coverage_csv",
    )))
    root, v21, v23, v33, cj, cc = roots
    if (base == root or base.is_relative_to(root)
            or any(freeze_path.is_relative_to(folder)
                   for folder in (root, v21, v23, v33))):
        raise ValueError("Provenance freeze stored in protected data roots")
    expected_partition_paths = {
        (v33 / f"symbol={symbol}" / f"month={LAST_MONTH}" / rel).resolve()
        for symbol in TARGETS for rel in KIND_FILES
    }
    actual_partition_paths = []
    for item in partitions:
        if set(item) != {"path", "sha256"}:
            raise ValueError("Unexpected V33 file inventory fields")
        path = Path(item["path"]).resolve()
        _check_hash("partition", item["sha256"])
        if (path not in expected_partition_paths or
                not path.is_file() or sha_file(path) != item["sha256"]):
            raise ValueError("Frozen V33 partition file SHA/identity mismatch: " + str(path))
        actual_partition_paths.append(path)
    if set(actual_partition_paths) != expected_partition_paths or len(set(actual_partition_paths)) != 56:
        raise ValueError("Incomplete/duplicate V33 original source files")
    for symbol in TARGETS:
        folder = v33 / f"symbol={symbol}" / f"month={LAST_MONTH}"
        found = {file.resolve() for file in folder.rglob("*") if file.is_file()}
        if found != {p for p in expected_partition_paths if p.is_relative_to(folder)}:
            raise ValueError("Unexpected/altered V33 partition file set")
    audit = json.loads(fixed[APPROVED_AUDIT_JSON_NAME].read_bytes())
    if (audit.get("protocol") != V33_AUDIT_PROTOCOL
            or audit.get("expected") != 84
            or audit.get("counts") != FROZEN_COUNTS
            or audit.get("total_verified") != 84
            or audit.get("no_network") is not True):
        raise ValueError("Frozen V33 audit result invalid")
    frame = pd.read_csv(fixed[APPROVED_AUDIT_CSV_NAME], dtype=str)
    if (len(frame) != 84 or frame[["symbol", "month"]].duplicated().any()
            or set(zip(frame.symbol, frame.month))
            != {(s, m) for s in SYMBOLS for m in EXPECTED_MONTHS}
            or frame.status.value_counts().to_dict()
            != {k:v for k,v in FROZEN_COUNTS.items() if v}):
        raise ValueError("Frozen V33 audit CSV mismatch")
    summary = json.loads(summary_file.read_bytes())
    if (summary.get("plan_sha256") != APPROVED_PLAN_SHA256
            or summary.get("journal_sha256") != sha_file(journal_file)
            or summary.get("newly_verified") != 4
            or summary.get("audit", {}).get("total_verified") != 84
            or summary.get("audit", {}).get("counts") != FROZEN_COUNTS
            or summary.get("failed") is not None
            or summary.get("final_audit_error") is not None):
        raise ValueError("Frozen batch journal/summary mismatch")
    return roots, plan, frame


class UnifiedDevelopmentReader:
    """Market/funding month reader. Construct only through preflight()."""

    def __init__(self, roots, identities):
        self._roots = roots
        self._identities = identities

    @classmethod
    def preflight(cls, freeze_path: Path, approved_sha: str):
        roots, plan, frozen_frame = _load_and_check_freeze(freeze_path, approved_sha)
        root, v21, v23, v33, cj, cc = roots
        current = audit_records(*roots)  # V24 full source replay + 4 V33 dual-source audits
        counts = audit_counts(current)
        if counts["total_verified"] != 84 or counts["counts"] != FROZEN_COUNTS:
            raise ValueError("84/84 replay verification failed")
        if {(r["symbol"], r["month"], r["status"]) for r in current} != {
                (row.symbol, row.month, row.status)
                for row in frozen_frame.itertuples(index=False)}:
            raise ValueError("Frozen V33 CSV differs from live verification")
        old = legacy_registry(root, v21, v23)
        if snapshot(old, v21, v23) != plan.get("verified_manifests"):
            raise ValueError("80 frozen V21/V23 manifest SHA-256 values changed")
        ids = {(r["symbol"], r["month"]): r["status"] for r in current}
        if len(ids) != 84:
            raise ValueError("Duplicate development partitions")
        return cls(roots, ids)

    def read_month(self, symbol: str, month: str):
        """Read only an approved, already fully verified development month.

        Return memory-only market and funding frames, never a price export.
        June 2026's final M5 bar closes at the OOS cutoff and is removed.
        """
        _approved_key(symbol, month)  # first: never touch an out-of-scope path
        status = self._identities.get((symbol, month))
        if status not in ("VERIFIED_V21", "VERIFIED_V23", "VERIFIED_V33"):
            raise ValueError("Requested partition lacks frozen verification")
        root, v21, v23, v33, cj, cc = self._roots
        if status == "VERIFIED_V21":
            folder = v21
        elif status == "VERIFIED_V23":
            folder = v23 / f"symbol={symbol}" / f"month={month}"
        else:
            if month != LAST_MONTH or symbol not in TARGETS:
                raise ValueError("V33 outside June target universe")
            folder = v33 / f"symbol={symbol}" / f"month={month}"
        manifest = json.loads((folder/"manifest.json").read_bytes())
        if manifest.get("symbol") != symbol or manifest.get("month") != month:
            raise ValueError("Unexpected partition manifest identity")
        if sha_file(folder/"markets_5m.parquet") != manifest["markets_sha256"]:
            raise ValueError("Market Parquet changed after provenance preflight")
        if sha_file(folder/"funding_events.parquet") != manifest["funding_sha256"]:
            raise ValueError("Funding Parquet changed after provenance preflight")
        markets = pd.read_parquet(folder/"markets_5m.parquet")
        funding = pd.read_parquet(folder/"funding_events.parquet")
        if list(markets.columns) != list(MARKET_FIELDS):
            raise ValueError("Unexpected normalized market schema")
        if list(funding.columns) != list(FUNDING_FIELDS):
            raise ValueError("Unexpected normalized funding schema")
        begin, end = _month_times(month)
        expected_count = int((end-begin)/M5)
        if len(markets) != expected_count or funding.empty:
            raise ValueError("Wrong market/funding month size")
        for col in ("bar_open_at", "bar_close_at"):
            markets[col] = pd.to_datetime(markets[col], utc=True)
        funding["funding_at"] = pd.to_datetime(funding["funding_at"], utc=True)
        opens = markets["bar_open_at"]
        closes = markets["bar_close_at"]
        expected_open = pd.date_range(begin, end-M5, freq="5min")
        if (not pd.DatetimeIndex(opens).equals(expected_open)
                or not closes.eq(opens + M5).all()):
            raise ValueError("Noncontiguous native M5 grid")
        if (not funding.funding_at.between(begin,end,inclusive="left").all()
                or funding.funding_at.duplicated().any()
                or not funding.funding_at.is_monotonic_increasing):
            raise ValueError("Funding event timestamps invalid")
        vals = markets[list(MARKET_FIELDS)[2:]].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("Nonfinite market values")
        if not funding[["funding_rate_fraction","funding_interval_hours"]].apply(
                pd.to_numeric, errors="raise").apply(np.isfinite).all().all():
            raise ValueError("Nonfinite funding values")
        if month == LAST_MONTH:
            crossing = closes.ge(CUTOFF)
            if (crossing.sum() != 1
                    or opens.loc[crossing].iloc[0] != CUTOFF-M5
                    or closes.loc[crossing].iloc[0] != CUTOFF):
                raise ValueError("Invalid June 2026 exclusive cutoff boundary")
            markets = markets.loc[~crossing].copy().reset_index(drop=True)
        elif closes.ge(CUTOFF).any():
            raise ValueError("Read attempted to cross held-out boundary")
        if (markets.empty or not markets.bar_close_at.lt(CUTOFF).all()
                or len(markets) != expected_count-(1 if month==LAST_MONTH else 0)):
            raise ValueError("Development bars cross reserved OOS boundary")
        return markets, funding, status


def readiness(reader: UnifiedDevelopmentReader):
    """Audit market/funding availability and time geometry; no economic calculations."""
    records=[]
    for symbol in SYMBOLS:
        previous_close = None
        for month in EXPECTED_MONTHS:
            market, fund, status = reader.read_month(symbol, month)
            first = market.bar_open_at.iloc[0]
            last = market.bar_close_at.iloc[-1]
            if previous_close is not None and first != previous_close:
                raise ValueError(f"{symbol}: cross-month M5 gap/overlap at {month}")
            previous_close = last
            records.append({
                "symbol": symbol, "month": month, "status": status,
                "market_m5_dev": int(len(market)),
                "funding_events_month": int(len(fund)),
                "first_open_utc": first.isoformat(),
                "last_close_utc": last.isoformat(),
                "excluded_cutoff_bar": int(month==LAST_MONTH),
            })
        selected = [r for r in records if r["symbol"]==symbol]
        if (len(selected)!=len(EXPECTED_MONTHS)
                or selected[0]["first_open_utc"] != BEGIN.isoformat()
                or previous_close != CUTOFF-M5):
            raise ValueError("Development history start/end mismatch for "+symbol)
    if len(records)!=84 or sum(r["excluded_cutoff_bar"] for r in records)!=4:
        raise ValueError("Unexpected development inventory")
    return records


def run(freeze: Path, freeze_sha256: str, output_json: Path, output_csv: Path):
    output_json=output_json.resolve()
    output_csv=output_csv.resolve()
    if (output_json==output_csv or output_json.exists() or output_csv.exists()
            or not output_json.parent.is_dir() or not output_csv.parent.is_dir()):
        raise ValueError("Require two distinct NEW audit reports")
    reader = UnifiedDevelopmentReader.preflight(freeze, freeze_sha256)
    root,v21,v23,v33,cj,cc = reader._roots
    for path in (output_json,output_csv):
        if (path==freeze.resolve()
                or any(path==folder or path.is_relative_to(folder)
                       for folder in (root,v21,v23,v33))):
            raise ValueError("Audit output overlaps protected partition roots")
    rows = readiness(reader)
    frame = pd.DataFrame(rows)
    protocol_counts = frame.status.value_counts().to_dict()
    if protocol_counts != {k:v for k,v in FROZEN_COUNTS.items() if v}:
        raise ValueError("Unknown provenance protocol counts")
    report = {
        "protocol": PROTOCOL,
        "frozen_v33_provenance_sha256": freeze_sha256,
        "calendar_months_per_symbol": len(EXPECTED_MONTHS),
        "universe": list(SYMBOLS),
        "expected_source_partitions": 84,
        "verified_source_partitions": len(rows),
        "by_protocol": protocol_counts,
        "market_m5_dev_per_symbol": {
            symbol:int(frame.loc[frame.symbol==symbol,"market_m5_dev"].sum())
            for symbol in SYMBOLS
        },
        "excluded_june_cutoff_bars": int(frame.excluded_cutoff_bar.sum()),
        "earliest_eligible_open_utc": BEGIN.isoformat(),
        "exclusive_holdout_cutoff_utc": CUTOFF.isoformat(),
        "status": "DEVELOPMENT_ONLY_84_VERIFIED_NO_HOLDOUT_ACCESS",
        "no_network_no_training_no_backtest_no_pnl_no_orders": True,
        "no_raw_price_export_no_imputation": True,
    }
    with output_csv.open("x",encoding="utf-8",newline="") as out:
        frame.to_csv(out,index=False)
    with output_json.open("x",encoding="utf-8") as out:
        json.dump(report,out,indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description="V34 offline unified development-readiness audit")
    parser.add_argument("--freeze-json",required=True,type=Path)
    parser.add_argument("--freeze-sha256",required=True,type=str)
    parser.add_argument("--output-json",required=True,type=Path)
    parser.add_argument("--output-csv",required=True,type=Path)
    a=parser.parse_args()
    report=run(a.freeze_json,a.freeze_sha256,a.output_json,a.output_csv)
    print("V34",report["status"],report["verified_source_partitions"],
          "source partitions",flush=True)
    print("PERIODS 2024-10..2026-06 / CUTOFF 2026-07-01 EXCLUSIVE",flush=True)
    print("NO NETWORK / NO HELD-OUT READ / NO TRAINING / NO PNL / NO ORDERS",flush=True)


if __name__ == "__main__":
    main()
