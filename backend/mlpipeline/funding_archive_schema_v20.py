"""V20: schema-only archive funding verification. Read-only, no returns or orders."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_archive_integrity_v19 import (
    BASE, PATHS, SYMBOL, MONTH, ZIP_LIMIT, UNCOMPRESSED_LIMIT,
    checksum, fetch, classify_timestamp,
)

PROTOCOL="p4_v20_funding_archive_schema_only_v1"
ALIASES=(("calc_time","last_funding_rate"),("fundingTime","fundingRate"))
FUNDING_STATES=("FUNDING_SCHEMA_AND_CHECKSUM_OK","SCHEMA_REVIEW_REQUIRED")


def verified_csv(blob:bytes)->bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members=archive.infolist()
        if len(members)!=1 or members[0].is_dir():
            raise ValueError("Expected one CSV")
        entry=members[0]
        if (not entry.filename.endswith(".csv") or
            "/" in entry.filename or "\\" in entry.filename):
            raise ValueError("Unsafe ZIP member")
        if (entry.file_size>UNCOMPRESSED_LIMIT or entry.compress_size==0 or
            entry.file_size / entry.compress_size>200):
            raise ValueError("Unsafe ZIP expansion")
        with archive.open(entry) as file:
            data=file.read(UNCOMPRESSED_LIMIT+1)
    if len(data)>UNCOMPRESSED_LIMIT:raise ValueError("Oversized expanded ZIP")
    return data


def funding_schema(data:bytes)->dict:
    frame=pd.read_csv(io.BytesIO(data),dtype=str,keep_default_na=False)
    columns=[str(x).strip() for x in frame.columns]
    if len(columns)!=len(set(columns)):
        raise ValueError("Duplicate funding columns")
    found=next(((time,rate) for time,rate in ALIASES
                if time in columns and rate in columns),None)
    if found is None:
        return {"status":"SCHEMA_REVIEW_REQUIRED","columns_observed":columns[:24],
                "count":len(frame)}
    if frame.empty:raise ValueError("Empty funding archive")
    time_col,rate_col=found
    raw=frame[time_col].str.strip()
    if not raw.str.fullmatch(r"\d{10,16}").all():
        raise ValueError("Invalid funding epoch")
    times,unit=classify_timestamp(raw)
    values=frame[rate_col].str.strip()
    if not values.str.fullmatch(r"-?(?:\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?").all():
        raise ValueError("Funding rate must be a decimal fraction")
    rates=pd.to_numeric(values,errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(rates).all() or (np.abs(rates)>0.1).any()):
        raise ValueError("Implausible funding rate")
    interval_column="funding_interval_hours" if "funding_interval_hours" in columns else None
    if interval_column:
        intervals=frame[interval_column].str.strip()
        if not intervals.str.fullmatch(r"\d+").all():
            raise ValueError("Invalid funding interval")
        numbers=pd.to_numeric(intervals,errors="raise")
        if not numbers.between(1,24,inclusive="both").all():
            raise ValueError("Funding interval out of bounds")
    return {"status":"FUNDING_SCHEMA_AND_CHECKSUM_OK",
            "columns_observed":columns,
            "count":len(frame),"time_field":time_col,"rate_field":rate_col,
            "timestamp_unit":unit,
            "first_time_utc":times.iloc[0].isoformat(),
            "last_time_utc":times.iloc[-1].isoformat(),
            "interval_field":interval_column,
            "no_future_rate_prediction":True}


def run(v19:Path)->dict:
    raw=v19.read_bytes()
    reference=json.loads(raw)
    if (reference.get("sample")!="BTCUSDT June 2025" or
        reference.get("protocol")!="p4_v19_btc_june2025_archive_integrity_pilot_v1" or
        any(reference.get("files",{}).get(k,{}).get("status")!="CHECKSUM_AND_SCHEMA_OK"
            for k in ("spot","perp","mark")) or
        any(reference.get("files",{}).get(k,{}).get("sha256_verified") is not True
            for k in ("spot","perp","mark")) or
        reference.get("files",{}).get("funding",{}).get("sha256_verified") is not True):
        raise ValueError("V19 reference not fully verified")
    url=BASE+PATHS["funding"].format(s=SYMBOL,m=MONTH)
    blob=fetch(url,ZIP_LIMIT)
    expected=checksum(fetch(url+".CHECKSUM",4096))
    actual=hashlib.sha256(blob).hexdigest()
    if actual!=expected:raise ValueError("Funding SHA256 checksum mismatch")
    details=funding_schema(verified_csv(blob))
    return {"protocol":PROTOCOL,"sample":reference["sample"],
            "v19_report_sha256":hashlib.sha256(raw).hexdigest(),
            "funding_zip_sha256":actual,
            "spot_perp_mark_alignment":reference.get("spot_perp_mark_alignment",{}),
            "funding":details,"no_trade_or_backtest":True,
            "coverage_of_21_months":"NOT_PROVEN"}


def main():
    parser=argparse.ArgumentParser(description="Validate one funding archive, no economics")
    parser.add_argument("--reference-v19-json",type=Path,required=True)
    parser.add_argument("--output-json",type=Path,required=True)
    parser.add_argument("--inspect-funding-archive",action="store_true")
    args=parser.parse_args()
    if not args.inspect_funding_archive:
        parser.error("Explicit --inspect-funding-archive is required for network access")
    source=args.reference_v19_json.resolve()
    output=args.output_json.resolve()
    if not source.is_file() or source==output or output.exists() or not output.parent.is_dir():
        parser.error("Existing V19 report and NEW output required")
    result=run(source)
    with output.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2,ensure_ascii=False)
    print("Funding:",result["funding"]["status"],"records",result["funding"].get("count"))
    print("Observed columns:",result["funding"]["columns_observed"])
    print("NO BACKTEST / NO TRADING / NO ASSUMED FULL COVERAGE")


if __name__=="__main__":
    main()
