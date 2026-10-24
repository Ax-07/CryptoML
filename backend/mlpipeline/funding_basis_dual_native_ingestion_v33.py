"""V33: fail-closed native dual-archive June 2026 funding/basis partitions.

V23 and V24 are intentionally unmodified. This module independently audits
V33 partitions from retained *original* monthly and daily mark archives.
Never writes P0/V21/V23, imputes candles, trades, backtests or trains.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_archive_integrity_v19 import (
    BASE, PATHS, ZIP_LIMIT, fetch, checksum,
)
from mlpipeline.funding_archive_schema_v20 import verified_csv
from mlpipeline.funding_basis_verified_partition_v23 import (
    M5, compare_p0, parse_funding, parse_kline, validate_coverage,
)
from mlpipeline.funding_basis_daily_mark_reconciliation_v30 import (
    DAILY_PATH, DAY_BEGIN, DAY_END, reconcile,
)
from mlpipeline.funding_basis_batch_resume_v25 import _paths as base_paths
from mlpipeline.funding_basis_controlled_batches_v27 import snapshot
from mlpipeline.funding_basis_partition_registry_v24 import registry

PROTOCOL = "p4_v33_immutable_dual_native_mark_june2026_v1"
PLAN_PROTOCOL = "p4_v33_approved_dual_source_plan_v1"
AUDIT_PROTOCOL = "p4_v33_extended_immutable_partition_registry_v1"
TARGETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MONTH = "2026-06"
MARK_DAY = "2026-06-29"
MIN_FREE = 512 * 1024 * 1024
KINDS = ("spot", "perp", "mark_monthly", "mark_daily", "funding")
MARK_PINNED = {
    "BTCUSDT": (
        "e02cf899d32a506fcb3165e795040dadea291989bdd12082bec1a83875d908fc",
        "eef21086c97542d63dcc88d311e9c3835d107f9d4b4497f6b488e32640e778c7",
    ),
    "ETHUSDT": (
        "de38736aea72c09d2802071b7c49bb108557fc5a5c81022bf8d706fd7e42430d",
        "b3bbf2991a983e822d860db0dda82383ed896deeb9195a74d7d25e6f524f3bb8",
    ),
    "SOLUSDT": (
        "0af68742b32f759dd96a0b93e945184ad85a61a630e09a16e9072174a46cfa55",
        "40fbc271c9b2b3d563fcf7570cb18ba14af600a602ad35735923abc0b55c06d7",
    ),
    "XRPUSDT": (
        "47c62563101585f9749763bca4b9b1813ac4ddfdcd1232dd074e54baa8244fcd",
        "dde8c6e6d338e9b2a05192d37b22a45c0064e1d292fa99997b4e87540b738e47",
    ),
}


# Compatibility alias for the frozen archive-pin contract tested by V33.
MARK_PINN = MARK_PINNED


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_roots(root, v21, v23, v33, coverage_json, coverage_csv):
    root, v21, v23, cj, cc = base_paths(
        root, v21, v23, coverage_json, coverage_csv
    )
    v33 = v33.resolve()
    if not v33.is_dir():
        raise ValueError("V33 output root must already exist")
    for protected in (root, v21, v23):
        if (v33 == protected or v33.is_relative_to(protected)
                or protected.is_relative_to(v33)):
            raise ValueError("V33 root overlaps immutable source roots")
    return root, v21, v23, v33, cj, cc


def new_report(path: Path, roots: tuple[Path, ...]):
    path = path.resolve()
    if not path.parent.is_dir() or path.exists():
        raise ValueError("Report must be NEW in an existing directory")
    if any(path == root or path.is_relative_to(root) for root in roots[:4]):
        raise ValueError("Report under protected or partition root")
    return path


def archive_url(symbol: str, kind: str) -> str:
    if symbol not in TARGETS:
        raise ValueError("Symbol outside frozen target universe")
    if kind == "mark_daily":
        return BASE + DAILY_PATH.replace("BTCUSDT", symbol)
    if kind == "mark_monthly":
        return BASE + PATHS["mark"].format(s=symbol, m=MONTH)
    if kind not in ("spot", "perp", "funding"):
        raise ValueError("Unknown source kind")
    return BASE + PATHS[kind].format(s=symbol, m=MONTH)


def check_reports(btc_path: Path, others_path: Path):
    btc = json.loads(btc_path.read_bytes())
    other = json.loads(others_path.read_bytes())
    if (btc.get("protocol") != "p4_v30_btc_june2026_daily_mark_reconciliation_v1"
            or btc.get("symbol") != "BTCUSDT"
            or other.get("protocol") != "p4_v32_remaining_june2026_daily_mark_reconciliation_v1"
            or other.get("all_source_candidates_eligible") is not True):
        raise ValueError("Wrong V30/V32 provenance report")
    if [r.get("symbol") for r in other.get("reports", [])] != list(TARGETS[1:]):
        raise ValueError("Unexpected source targets in V32 report")
    for symbol, item in zip(TARGETS, [btc] + other["reports"]):
        status = item.get("reconciliation", {})
        observed_day = (item.get("daily_source_date") if symbol == "BTCUSDT"
                        else item.get("day"))
        if (item.get("month") != MONTH or observed_day != MARK_DAY
                or status.get("status") != "SOURCE_RECONCILIATION_CANDIDATE_NOT_PUBLISHED"
                or status.get("timeline_reconciliation_eligible_for_review") is not True
                or status.get("monthly_missing_count") != 288
                or status.get("missing_exactly_2026_06_29") is not True
                or status.get("daily_m5_count") != 288
                or status.get("monthly_daily_overlap_count") != 0
                or status.get("combined_timeline_complete") is not True):
            raise ValueError("Invalid V30/V32 reconciliation: " + symbol)
        for field, pin, kind in (
            ("monthly_archive", MARK_PINNED[symbol][0], "mark_monthly"),
            ("daily_archive", MARK_PINNED[symbol][1], "mark_daily"),
        ):
            source = item.get(field, {})
            if (source.get("url") != archive_url(symbol, kind)
                    or source.get("zip_sha256") != pin
                    or source.get("checksum_verified") is not True):
                raise ValueError("Mark ZIP origin/SHA changed: " + symbol)
    return {"v30_sha256": sha_file(btc_path), "v32_sha256": sha_file(others_path)}


def mark_rows(data: bytes) -> pd.DataFrame:
    f = pd.read_csv(io.BytesIO(data), header=None, dtype=str,
                    keep_default_na=False, low_memory=False)
    if f.empty:
        raise ValueError("Empty mark CSV")
    if not f.iat[0, 0].strip().isdigit():
        f = f.iloc[1:].reset_index(drop=True)
    if f.empty or f.shape[1] != 12:
        raise ValueError("Incorrect mark CSV shape")
    return f


def reconstruct_mark(monthly: bytes, daily: bytes):
    details = reconcile(monthly, daily)
    if not details["timeline_reconciliation_eligible_for_review"]:
        raise ValueError("Mark native reconciliation refused")
    if details["monthly_epoch_unit"] != "ms" or details["daily_epoch_unit"] != "ms":
        raise ValueError("Unexpected mark epoch unit")
    a, b = mark_rows(monthly), mark_rows(daily)
    if len(a) != 8352 or len(b) != 288:
        raise ValueError("Mark source counts changed")
    f = pd.concat([a, b], ignore_index=True)
    raw_time = pd.to_numeric(f.iloc[:, 0], errors="raise").astype("int64")
    f = f.iloc[np.argsort(raw_time.to_numpy(), kind="stable")]
    merged_csv = f.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
    begin = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-07-01T00:00:00Z")
    return merged_csv, parse_kline(merged_csv, "mark", begin, end), details


def verify_raw_zip(zip_bytes: bytes, checksum_bytes: bytes, pin=None):
    actual = sha_bytes(zip_bytes)
    if actual != checksum(checksum_bytes):
        raise ValueError("Original Binance ZIP/CHECKSUM mismatch")
    if pin is not None and actual != pin:
        raise ValueError("ZIP differs from approved SHA")
    return {
        "zip_sha256": actual,
        "checksum_sha256": sha_bytes(checksum_bytes),
    }


# Compatibility alias: identical immutable SHA/checksum validation semantics.
checked_zip_blob = verify_raw_zip


def decode_original(raw_dir: Path, symbol: str, download=False):
    sources = {}
    csv = {}
    for kind in KINDS:
        url = archive_url(symbol, kind)
        if download:
            blob = fetch(url, ZIP_LIMIT)
            proof = fetch(url + ".CHECKSUM", 4096)
        else:
            blob = (raw_dir / (kind + ".zip")).read_bytes()
            proof = (raw_dir / (kind + ".zip.CHECKSUM")).read_bytes()
        pin = (MARK_PINNED[symbol][0] if kind == "mark_monthly" else
               MARK_PINNED[symbol][1] if kind == "mark_daily" else None)
        hashes = verify_raw_zip(blob, proof, pin)
        csv[kind] = verified_csv(blob)
        sources[kind] = {"url": url, **hashes}
        if download:
            (raw_dir / (kind + ".zip")).write_bytes(blob)
            (raw_dir / (kind + ".zip.CHECKSUM")).write_bytes(proof)
            print(kind, "SHA VERIFIED", flush=True)
    begin = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-07-01T00:00:00Z")
    spot = parse_kline(csv["spot"], "spot", begin, end)
    perp = parse_kline(csv["perp"], "perp", begin, end)
    merged_csv, mark, details = reconstruct_mark(csv["mark_monthly"], csv["mark_daily"])
    funding = parse_funding(csv["funding"], begin, end)
    for item in (perp, mark):
        if not item[0].reset_index(drop=True).equals(spot[0].reset_index(drop=True)):
            raise ValueError("Markets' UTC timelines differ")
    sources["spot"]["timestamp_unit"] = spot[1]
    sources["perp"]["timestamp_unit"] = perp[1]
    sources["mark_monthly"]["timestamp_unit"] = details["monthly_epoch_unit"]
    sources["mark_daily"]["timestamp_unit"] = details["daily_epoch_unit"]
    return (spot, perp, mark, funding), merged_csv, sources


def market_frame(decoded):
    spot, perp, mark, _funding = decoded
    frame = pd.DataFrame({"bar_open_at": spot[0], "bar_close_at": spot[0] + M5})
    for kind, item in (("spot", spot), ("perp", perp), ("mark", mark)):
        for name, values in item[2].items():
            frame[kind + "_" + name] = values
    return frame


def build_one(symbol, root, v33, coverage_json, coverage_csv, plan):
    dest = v33 / ("symbol=" + symbol) / ("month=" + MONTH)
    if dest.exists():
        raise ValueError("Existing destination must never be overwritten")
    if shutil.disk_usage(v33).free < MIN_FREE:
        raise ValueError("Not enough free disk space")
    lineage = validate_coverage(coverage_json, coverage_csv)
    if lineage != plan["v22"]:
        raise ValueError("V22 immutable coverage changed")
    with tempfile.TemporaryDirectory(dir=v33, prefix="v33-staging-") as tmp:
        stage = Path(tmp)
        raw = stage / "raw"
        raw.mkdir()
        parsed, mark_csv, origins = decode_original(raw, symbol, download=True)
        p0sha = compare_p0(root, symbol, MONTH, parsed[1][0], parsed[1][2])
        markets = market_frame(parsed)
        funding = parsed[3]
        mp, fp, cp = (
            stage / "markets_5m.parquet",
            stage / "funding_events.parquet",
            stage / "mark_reconstructed.csv",
        )
        markets.to_parquet(mp, index=False, engine="pyarrow", compression="zstd")
        funding.to_parquet(fp, index=False, engine="pyarrow", compression="zstd")
        cp.write_bytes(mark_csv)
        manifest = {
            "protocol": PROTOCOL,
            "symbol": symbol,
            "month": MONTH,
            "source_plan_sha256": plan["approved_plan_sha256"],
            "reports_sha256": plan["reports_sha256"],
            "v22": lineage,
            "p0_sha256": p0sha,
            "archives": origins,
            "mark_reconciliation": {
                "native_monthly_rows": 8352,
                "native_daily_rows": 288,
                "day": MARK_DAY,
                "imputed_rows": 0,
                "provenance": "TWO_CHECKSUM_VERIFIED_NATIVE_BINANCE_MARK_ARCHIVES",
            },
            "market_bars": len(markets),
            "funding_events": len(funding),
            "mark_reconstructed_sha256": sha_file(cp),
            "markets_sha256": sha_file(mp),
            "funding_sha256": sha_file(fp),
            "status": "DUAL_NATIVE_SOURCE_CHECKSUM_AND_P0_VERIFIED_NO_PNL",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                              encoding="utf-8")
        # Full source replay and P0 verification must succeed *before* publication.
        # A rejected stage is automatically discarded by TemporaryDirectory,
        # never leaving a FAILED_INTEGRITY target directory in the registry.
        verify_one(stage, symbol, root, coverage_json, coverage_csv)
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            raise ValueError("Destination appeared before atomic rename")
        stage.rename(dest)
    return dest


def verify_one(folder, symbol, root, cj, cc):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("protocol") != PROTOCOL
            or manifest.get("symbol") != symbol
            or manifest.get("month") != MONTH
            or manifest.get("status") != "DUAL_NATIVE_SOURCE_CHECKSUM_AND_P0_VERIFIED_NO_PNL"
            or manifest.get("v22") != validate_coverage(cj, cc)):
        raise ValueError("Invalid V33 identity or immutable lineage")
    data, mark_csv, origins = decode_original(folder / "raw", symbol)
    if origins != manifest.get("archives"):
        raise ValueError("Stored source origins do not match raw archives")
    if manifest.get("mark_reconciliation") != {
        "native_monthly_rows": 8352,
        "native_daily_rows": 288,
        "day": MARK_DAY,
        "imputed_rows": 0,
        "provenance": "TWO_CHECKSUM_VERIFIED_NATIVE_BINANCE_MARK_ARCHIVES",
    }:
        raise ValueError("Mark native-source provenance changed")
    if sha_bytes(mark_csv) != manifest.get("mark_reconstructed_sha256"):
        raise ValueError("Reconstructed mark hash mismatch")
    if sha_file(folder / "mark_reconstructed.csv") != sha_bytes(mark_csv):
        raise ValueError("Stored reconstructed mark differs from source replay")
    mp, fp = folder / "markets_5m.parquet", folder / "funding_events.parquet"
    if (sha_file(mp) != manifest.get("markets_sha256")
            or sha_file(fp) != manifest.get("funding_sha256")):
        raise ValueError("Parquet digest mismatch")
    markets = pd.read_parquet(mp)
    expected = market_frame(data)
    if list(markets.columns) != list(expected.columns) or len(markets) != 8640:
        raise ValueError("Market output has invalid shape")
    for name in expected.columns[:2]:
        a = pd.to_datetime(markets[name], utc=True).reset_index(drop=True)
        b = pd.to_datetime(expected[name], utc=True).reset_index(drop=True)
        if not a.equals(b):
            raise ValueError("Market timestamp mismatch: " + name)
    for name in expected.columns[2:]:
        if not np.array_equal(markets[name].to_numpy(dtype=float),
                              expected[name].to_numpy(dtype=float)):
            raise ValueError("Source replay mismatch: " + name)
    funding = pd.read_parquet(fp)
    original_funding = data[3]
    if (list(funding.columns) != list(original_funding.columns)
            or len(funding) != len(original_funding)
            or len(funding) != manifest.get("funding_events") or funding.empty):
        raise ValueError("Funding output has invalid shape")
    if not pd.to_datetime(funding["funding_at"],utc=True).reset_index(drop=True).equals(
            pd.to_datetime(original_funding["funding_at"],utc=True).reset_index(drop=True)):
        raise ValueError("Funding timestamps differ from native source")
    for name in ("funding_rate_fraction", "funding_interval_hours"):
        if not np.array_equal(pd.to_numeric(funding[name]).to_numpy(),
                              pd.to_numeric(original_funding[name]).to_numpy()):
            raise ValueError("Funding source replay mismatch: " + name)
    p0 = compare_p0(root, symbol, MONTH, data[1][0], data[1][2])
    if p0 != manifest.get("p0_sha256"):
        raise ValueError("P0 digest mismatch")
    if manifest.get("market_bars") != 8640:
        raise ValueError("Incorrect manifest bar count")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_plan_sha256", "")):
        raise ValueError("Invalid operator plan digest")
    return {"symbol":symbol,"month":MONTH,"status":"VERIFIED_V33","m5":8640}


def audit_records(root,v21,v23,v33,cj,cc):
    old = registry(root,v21,v23)
    if len(old)!=84 or old.status.eq("FAILED_INTEGRITY").any():
        raise ValueError("Protected V24 registry has integrity failure")
    result = []
    for row in old.itertuples(index=False):
        status = row.status
        if row.month == MONTH and row.symbol in TARGETS:
            if status != "MISSING":
                raise ValueError("V33 target unexpectedly exists in protected V23")
            folder = v33 / ("symbol=" + row.symbol) / ("month=" + MONTH)
            if folder.exists():
                try:
                    verify_one(folder,row.symbol,root,cj,cc)
                    status = "VERIFIED_V33"
                except (ValueError,OSError,KeyError,TypeError,IndexError,
                        json.JSONDecodeError) as exc:
                    result.append({"symbol":row.symbol,"month":row.month,
                                   "status":"FAILED_INTEGRITY",
                                   "error_type":type(exc).__name__})
                    continue
        result.append({"symbol":row.symbol,"month":row.month,
                       "status":status,"error_type":row.error_type})
    if len(result)!=84 or len(set((r["symbol"],r["month"]) for r in result))!=84:
        raise ValueError("Invalid frozen partition universe")
    return result


def audit_counts(rows):
    keys=("VERIFIED_V21","VERIFIED_V23","VERIFIED_V33","MISSING","FAILED_INTEGRITY")
    counts={name:sum(row["status"]==name for row in rows) for name in keys}
    return {"expected":84,"counts":counts,
            "total_verified":sum(counts[x] for x in keys[:3])}


def plan_sources(root,v21,v23,v33,cj,cc,btc_report,others_report,output):
    roots=checked_roots(root,v21,v23,v33,cj,cc)
    root,v21,v23,v33,cj,cc=roots
    output=new_report(output,roots)
    btc_report,others_report=btc_report.resolve(),others_report.resolve()
    if not btc_report.is_file() or not others_report.is_file():
        raise ValueError("Require both V30 and V32 reports")
    reports=check_reports(btc_report,others_report)
    rows=audit_records(*roots)
    if audit_counts(rows)["counts"] != {
        "VERIFIED_V21":1,"VERIFIED_V23":79,"VERIFIED_V33":0,
        "MISSING":4,"FAILED_INTEGRITY":0,
    }:
        raise ValueError("Preflight must be 80 verified and 4 missing")
    previous=registry(root,v21,v23)
    plan={
        "protocol":PLAN_PROTOCOL,
        "sources":dict(zip((
            "data_root","v21_directory","v23_root","v33_root",
            "coverage_json","coverage_csv","v30_report","v32_report",
        ),map(str,(*roots,btc_report,others_report)))),
        "targets":[{"symbol":s,"month":MONTH} for s in TARGETS],
        "verified_manifests":snapshot(previous,v21,v23),
        "v22":validate_coverage(cj,cc),
        "reports_sha256":reports,
        "initial_verified":80,
        "initial_missing":4,
        "max_new":4,
        "min_free_bytes":MIN_FREE,
        "no_pnl_no_training_no_orders":True,
    }
    with output.open("x",encoding="utf-8") as f:json.dump(plan,f,indent=2)
    return plan


def read_plan(path, approved_sha):
    digest=sha_file(path)
    if digest!=approved_sha.lower():
        raise ValueError("Operator plan SHA256 approval mismatch")
    data=json.loads(path.read_bytes())
    if (data.get("protocol")!=PLAN_PROTOCOL or
            data.get("targets")!=[{"symbol":s,"month":MONTH} for s in TARGETS] or
            data.get("initial_verified")!=80 or data.get("initial_missing")!=4 or
            data.get("max_new")!=4 or data.get("min_free_bytes")!=MIN_FREE or
            data.get("no_pnl_no_training_no_orders") is not True):
        raise ValueError("Invalid plan format or frozen target set")
    s=data["sources"]
    roots=checked_roots(*(Path(s[k]) for k in (
        "data_root","v21_directory","v23_root","v33_root",
        "coverage_json","coverage_csv",
    )))
    root,v21,v23,v33,cj,cc=roots
    if data["v22"]!=validate_coverage(cj,cc):
        raise ValueError("V22 lineage changed")
    rb,ro=Path(s["v30_report"]),Path(s["v32_report"])
    if data["reports_sha256"]!=check_reports(rb,ro):
        raise ValueError("Source audit provenance changed")
    old=registry(root,v21,v23)
    if old.status.eq("FAILED_INTEGRITY").any() or snapshot(old,v21,v23)!=data["verified_manifests"]:
        raise ValueError("Previously verified immutable partition changed")
    data["approved_plan_sha256"]=digest
    return data,roots


def execute(plan_path, approved_sha, journal, summary, allow=False):
    if not allow:raise ValueError("Explicit --permit-downloads required")
    plan_path=plan_path.resolve()
    data,roots=read_plan(plan_path,approved_sha)
    root,v21,v23,v33,cj,cc=roots
    journal=new_report(journal,roots)
    summary=new_report(summary,roots)
    if len({plan_path,journal,summary})!=3:
        raise ValueError("Plan/journal/summary must be distinct")
    records=audit_records(*roots)
    status={r["symbol"]:r["status"] for r in records if r["month"]==MONTH}
    if any(status[s] not in ("MISSING","VERIFIED_V33") for s in TARGETS):
        raise ValueError("Target failed pre-ingestion audit")
    completed=0
    failure=None
    with journal.open("x",encoding="utf-8") as f:
        def emit(event,symbol=None,detail=None):
            entry={"event":event,"symbol":symbol,"month":MONTH if symbol else None}
            if detail is not None:entry["detail"]=detail
            f.write(json.dumps(entry)+"\n")
            f.flush()
            os.fsync(f.fileno())
        emit("PRECHECK_OK")
        for symbol in TARGETS:
            if status[symbol]=="VERIFIED_V33":
                emit("ALREADY_VERIFIED_SKIPPED",symbol)
                continue
            emit("STARTED",symbol)
            try:
                folder=build_one(symbol,root,v33,cj,cc,data)
                verify_one(folder,symbol,root,cj,cc)
                completed+=1
                emit("VERIFIED",symbol,{"manifest_sha256":sha_file(folder/"manifest.json")})
            except Exception as exc:
                failure={"symbol":symbol,"month":MONTH,
                         "error_type":type(exc).__name__,"message":str(exc)[:260]}
                emit("FAILED",symbol,failure)
                break
        emit("BATCH_FAILED" if failure else "BATCH_DONE")
    try:
        counts=audit_counts(audit_records(*roots))
        audit_error=None
    except Exception as exc:
        counts=None
        audit_error={"type":type(exc).__name__,"message":str(exc)[:260]}
    result={
        "protocol":PROTOCOL,
        "plan_sha256":sha_file(plan_path),"journal_sha256":sha_file(journal),
        "newly_verified":completed,"audit":counts,
        "failed":failure,"final_audit_error":audit_error,
        "no_pnl_no_training_no_orders":True,
    }
    with summary.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    if failure or audit_error or counts["counts"]["FAILED_INTEGRITY"]:
        raise RuntimeError("V33 halted; preserve original journal and summary")
    return result


def main():
    p=argparse.ArgumentParser(description="V33 controlled dual-native-mark ingest and offline audit")
    p.add_argument("--mode",choices=("plan","execute","audit"),required=True)
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--plan-sha256")
    for key in (
        "data-root","v21-directory","v23-root","v33-root",
        "coverage-json","coverage-csv","v30-report","v32-report",
        "output-plan","plan-json","journal-jsonl","summary-json",
        "output-json","output-csv",
    ):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.mode=="plan":
        required=(a.data_root,a.v21_directory,a.v23_root,a.v33_root,
                  a.coverage_json,a.coverage_csv,a.v30_report,a.v32_report,
                  a.output_plan)
        if a.permit_downloads or any(x is None for x in required):
            p.error("Offline plan requires paths, reports and NEW output")
        plan=plan_sources(*required)
        print("V33 PLAN",plan["targets"],"initial",plan["initial_verified"])
        print("NO NETWORK / NO PNL / NO ORDERS")
    elif a.mode=="execute":
        if not all((a.permit_downloads,a.plan_json,a.plan_sha256,
                    a.journal_jsonl,a.summary_json)):
            p.error("Execute requires approval hash and NEW journal/summary")
        summary=execute(a.plan_json,a.plan_sha256,a.journal_jsonl,a.summary_json,True)
        print("V33",summary["audit"]["total_verified"],"verified",
              summary["audit"]["counts"]["MISSING"],"remaining")
        print("NO PNL / NO ORDERS")
    else:
        required=(a.data_root,a.v21_directory,a.v23_root,a.v33_root,
                  a.coverage_json,a.coverage_csv,a.output_json,a.output_csv)
        if a.permit_downloads or any(x is None for x in required):
            p.error("Offline audit requires all roots and NEW JSON/CSV")
        roots=checked_roots(*required[:6])
        jout=new_report(a.output_json,roots)
        cout=new_report(a.output_csv,roots)
        if jout==cout:raise ValueError("JSON and CSV reports must differ")
        rows=audit_records(*roots)
        report={"protocol":AUDIT_PROTOCOL,**audit_counts(rows),
                "no_network":True,"no_pnl_no_training_no_orders":True}
        with cout.open("x",encoding="utf-8") as f:
            pd.DataFrame(rows).to_csv(f,index=False)
        with jout.open("x",encoding="utf-8") as f:
            json.dump(report,f,indent=2)
        print("V33 AUDIT",report["counts"],"verified",report["total_verified"])
        print("NO NETWORK / NO PNL / NO ORDERS")


if __name__=="__main__":
    main()
