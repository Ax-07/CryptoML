"""V24 audited immutable partition registry; optional bounded V23 ingestion."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from mlpipeline.pullback_long_history_v3 import SYMBOLS,EXPECTED_MONTHS
from mlpipeline.funding_basis_verified_partition_v23 import ingest,compare_p0,PROTOCOL as V23_PROTOCOL
from mlpipeline.funding_basis_immutable_pilot_v21 import PROTOCOL as V21_PROTOCOL
from mlpipeline.funding_basis_archive_integrity_v19 import checksum

KINDS=("spot","perp","mark","funding")
PROTOCOL="p4_v24_immutable_partition_registry_v1"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def verify(folder,symbol,month,root,legacy=False):
    m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if (m.get("protocol")!=(V21_PROTOCOL if legacy else V23_PROTOCOL) or
        m.get("symbol")!=symbol or m.get("month")!=month or
        not isinstance(m.get("archives"),dict)):
        raise ValueError("Wrong manifest identity")
    for kind in KINDS:
        archive=m["archives"][kind]
        zipfile=folder/"raw"/f"{kind}.zip"
        checkfile=folder/"raw"/f"{kind}.zip.CHECKSUM"
        if (sha(zipfile)!=archive["zip_sha256"] or
            sha(checkfile)!=archive["checksum_sha256"] or
            checksum(checkfile.read_bytes())!=sha(zipfile)):
            raise ValueError("ZIP/CHECKSUM SHA mismatch: "+kind)
    markets_file=folder/"markets_5m.parquet"
    funding_file=folder/"funding_events.parquet"
    markets_sha=m["markets_sha256"]
    funding_sha=m["funding_sha256"]
    if sha(markets_file)!=markets_sha or sha(funding_file)!=funding_sha:
        raise ValueError("Parquet SHA mismatch")
    x=pd.read_parquet(markets_file)
    f=pd.read_parquet(funding_file)
    begin=pd.Timestamp(month+"-01T00:00:00Z")
    end=begin+pd.offsets.MonthBegin(1)
    expected=int((end-begin)/pd.Timedelta(minutes=5))
    if (len(x)!=expected or len(x)!=m["market_bars"] or
        len(f)!=m["funding_events"] or f.empty):
        raise ValueError("Incorrect partition row counts")
    opens=pd.to_datetime(x["bar_open_at"],utc=True)
    closes=pd.to_datetime(x["bar_close_at"],utc=True)
    if (opens.iloc[0]!=begin or opens.iloc[-1]!=end-pd.Timedelta(minutes=5) or
        not opens.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all() or
        not closes.eq(opens+pd.Timedelta(minutes=5)).all()):
        raise ValueError("Broken M5 UTC grid")
    for kind in ("spot","perp","mark"):
        names=[f"{kind}_{name}" for name in ("open","high","low","close")]
        prices=x[names].to_numpy(dtype=float)
        if (not np.isfinite(prices).all() or (prices<=0).any() or
            (prices[:,1]<prices.max(axis=1)).any() or
            (prices[:,2]>prices.min(axis=1)).any()):
            raise ValueError("Invalid OHLC "+kind)
    for kind in ("spot","perp"):
        volume=x[f"{kind}_volume"].to_numpy(dtype=float)
        if not np.isfinite(volume).all() or (volume<0).any():
            raise ValueError("Invalid volume "+kind)
    if "mark_volume" in x:raise ValueError("Fictitious mark volume")
    t=pd.to_datetime(f["funding_at"],utc=True)
    rates=pd.to_numeric(f["funding_rate_fraction"]).to_numpy(dtype=float)
    hours=pd.to_numeric(f["funding_interval_hours"]).to_numpy(dtype=float)
    if (not t.between(begin,end,inclusive="left").all() or
        t.duplicated().any() or not t.is_monotonic_increasing or
        not np.isfinite(rates).all() or (np.abs(rates)>0.1).any() or
        not np.isfinite(hours).all() or (hours!=np.floor(hours)).any() or
        ((hours<1)|(hours>24)).any()):
        raise ValueError("Invalid funding events")
    perp={key:x[f"perp_{key}"].to_numpy(dtype=float) for key in ("open","high","low","close")}
    p0sha=compare_p0(root,symbol,month,opens,perp)
    if p0sha!=m["p0_perp_parquet_sha256" if legacy else "p0_sha256"]:
        raise ValueError("P0 digest differs from source manifest")
    return {"m5":len(x),"funding_events":len(f),"p0_sha256":p0sha}


def registry(root,v21,v23):
    rows=[]
    for symbol in SYMBOLS:
        for month in EXPECTED_MONTHS:
            legacy=symbol=="BTCUSDT" and month=="2025-06"
            folder=(v21 if legacy else v23/f"symbol={symbol}"/f"month={month}")
            if not folder.exists():
                rows.append({"symbol":symbol,"month":month,"status":"MISSING","m5":None,
                             "funding_events":None,"error_type":None})
                continue
            try:
                data=verify(folder,symbol,month,root,legacy=legacy)
                status="VERIFIED_V21" if legacy else "VERIFIED_V23"
                rows.append({"symbol":symbol,"month":month,"status":status,
                             "m5":data["m5"],"funding_events":data["funding_events"],
                             "error_type":None})
            except (ValueError,OSError,KeyError,TypeError,IndexError, json.JSONDecodeError) as exc:
                rows.append({"symbol":symbol,"month":month,"status":"FAILED_INTEGRITY",
                             "m5":None,"funding_events":None,"error_type":type(exc).__name__})
    return pd.DataFrame(rows)


def write_report(frame,jsonpath,csvpath):
    if jsonpath==csvpath or any(x.exists() or not x.parent.is_dir() for x in (jsonpath,csvpath)):
        raise ValueError("Two NEW report files are required")
    if len(frame)!=84 or frame.duplicated(["symbol","month"]).any():
        raise AssertionError("Invalid 84-partition ledger")
    counts={k:int(frame.status.eq(k).sum()) for k in
            ("VERIFIED_V21","VERIFIED_V23","MISSING","FAILED_INTEGRITY")}
    report={"protocol":PROTOCOL,"expected":84,"counts":counts,"total_verified":
            counts["VERIFIED_V21"]+counts["VERIFIED_V23"],
            "no_network_in_audit":True,"no_pnl_no_training_no_orders":True}
    with csvpath.open("x",encoding="utf-8",newline="") as out:frame.to_csv(out,index=False)
    with jsonpath.open("x",encoding="utf-8") as out:json.dump(report,out,indent=2)
    return report


def main():
    p=argparse.ArgumentParser(description="V24 offline audit or opt-in max-two partition ingestion")
    p.add_argument("--mode",choices=("audit","ingest"),default="audit")
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--symbol",choices=SYMBOLS)
    p.add_argument("--start-month",choices=EXPECTED_MONTHS)
    p.add_argument("--max-new-partitions",type=int,default=1)
    for name in ("data-root","v21-directory","v23-root","coverage-json","coverage-csv",
                 "output-json","output-csv"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve();v21=a.v21_directory.resolve();v23=a.v23_root.resolve()
    jsonpath=a.output_json.resolve();csvpath=a.output_csv.resolve()
    if (jsonpath==csvpath or any(x.exists() or not x.parent.is_dir() or
        x.is_relative_to(root) for x in (jsonpath,csvpath)) or
        root==v23 or v23.is_relative_to(root) or root.is_relative_to(v23) or
        root==v21 or v21.is_relative_to(root) or not v21.is_dir() or not v23.is_dir()):
        p.error("Unsafe paths / reports must be new and outside P0")
    if a.mode=="audit" and (a.permit_downloads or a.symbol or a.start_month):
        p.error("Audit has no downloads or ingestion target")
    if a.mode=="ingest" and (not a.permit_downloads or not a.symbol or not a.start_month
                              or a.max_new_partitions not in (1,2)):
        p.error("Ingest requires explicit permission, asset, start month and budget 1 or 2")
    frame=registry(root,v21,v23)
    if frame.status.eq("FAILED_INTEGRITY").any() and a.mode=="ingest":
        raise ValueError("FAILED_INTEGRITY detected; stopping BEFORE downloads")
    if a.mode=="ingest":
        candidates=frame[(frame.symbol==a.symbol)&(frame.month>=a.start_month)&
                         frame.status.eq("MISSING")].head(a.max_new_partitions)
        if candidates.empty:print("No new eligible partition; existing data untouched")
        for r in candidates.itertuples(index=False):
            print("V24 ingest",r.symbol,r.month,flush=True)
            ingest(r.symbol,r.month,root,v23,a.coverage_json.resolve(),a.coverage_csv.resolve())
        frame=registry(root,v21,v23)
    report=write_report(frame,jsonpath,csvpath)
    print("V24",report["counts"],"verified",report["total_verified"])
    print("NO PNL / NO ORDERS / NO OOS CLAIM")


if __name__=="__main__":main()
