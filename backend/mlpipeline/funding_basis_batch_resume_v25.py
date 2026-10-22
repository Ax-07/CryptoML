"""V25: offline immutable microbatch plan; opt-in sequential, journaled V23 ingestion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

from mlpipeline.funding_basis_partition_registry_v24 import registry, verify
from mlpipeline.funding_basis_verified_partition_v23 import ingest, validate_coverage
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS

PROTOCOL="p4_v25_bounded_ingestion_plan_v1"
TARGETS=(("ETHUSDT","2024-10"),("SOLUSDT","2024-10"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(root,v21,v23,coverage_json,coverage_csv):
    sources=(root.resolve(),v21.resolve(),v23.resolve(),coverage_json.resolve(),coverage_csv.resolve())
    root,v21,v23,cj,cc=sources
    if (not root.is_dir() or not v21.is_dir() or not v23.is_dir() or
        not cj.is_file() or not cc.is_file() or
        root==v21 or root==v23 or
        v21.is_relative_to(root) or v23.is_relative_to(root) or
        root.is_relative_to(v21) or root.is_relative_to(v23) or
        v21==v23 or v21.is_relative_to(v23) or v23.is_relative_to(v21)):
        raise ValueError("Unsafe or missing source roots")
    return sources


def audit(root,v21,v23):
    frame=registry(root,v21,v23)
    if frame.status.eq("FAILED_INTEGRITY").any():
        raise ValueError("FAILED_INTEGRITY: repair manually before any download")
    return frame


def plan(root,v21,v23,coverage_json,coverage_csv,out):
    root,v21,v23,cj,cc=_paths(root,v21,v23,coverage_json,coverage_csv)
    out=out.resolve()
    if (out.exists() or not out.parent.is_dir() or
        any(out.is_relative_to(folder) for folder in (root,v21,v23))):
        raise ValueError("Plan output must be NEW and outside data roots")
    frame=audit(root,v21,v23)
    lineage=validate_coverage(cj,cc)
    entries=[]
    for symbol,month in TARGETS:
        found=frame[(frame.symbol==symbol)&(frame.month==month)]
        if len(found)!=1 or found.status.iloc[0]!="MISSING":
            raise ValueError("Predeclared candidate is not missing: "+symbol+" "+month)
        entries.append({"symbol":symbol,"month":month})
    report={"protocol":PROTOCOL,"sources":{
        "data_root":str(root),"v21_directory":str(v21),"v23_root":str(v23),
        "coverage_json":str(cj),"coverage_csv":str(cc)},
        "v22":lineage,"initial_verified":int(frame.status.isin(("VERIFIED_V21","VERIFIED_V23")).sum()),
        "initial_missing":int(frame.status.eq("MISSING").sum()),
        "targets":entries,"max_new":2,"no_economics":True}
    with out.open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    return report


def execute(plan_path,journal_path,summary_path,permit_downloads=False):
    if not permit_downloads:raise ValueError("Explicit --permit-downloads required")
    plan_path=plan_path.resolve();journal_path=journal_path.resolve();summary_path=summary_path.resolve()
    if (journal_path==summary_path or any(p.exists() or not p.parent.is_dir()
        for p in (journal_path,summary_path))):
        raise ValueError("Journal and summary must be NEW distinct files")
    data=json.loads(plan_path.read_bytes())
    if data.get("protocol")!=PROTOCOL or data.get("targets")!=[
        {"symbol":s,"month":m} for s,m in TARGETS] or data.get("max_new")!=2:
        raise ValueError("Plan protocol/target mismatch")
    a=data["sources"]
    root,v21,v23,cj,cc=_paths(*(Path(a[k]) for k in (
        "data_root","v21_directory","v23_root","coverage_json","coverage_csv")))
    if any(p.is_relative_to(folder) for p in (journal_path,summary_path)
           for folder in (root,v21,v23)):
        raise ValueError("Unsafe journal path")
    if validate_coverage(cj,cc)!=data["v22"]:
        raise ValueError("V22 coverage changed since planning")
    frame=audit(root,v21,v23)
    statuses={(r.symbol,r.month):r.status for r in frame.itertuples(index=False)}
    if any(statuses[t] not in ("MISSING","VERIFIED_V23") for t in TARGETS):
        raise ValueError("Unsafe target status")
    finished=0;failed=None
    # An exclusive creation protects existing journal data against accidental overwrite.
    with journal_path.open("x",encoding="utf-8") as log:
        def write(kind,symbol=None,month=None,error=None):
            entry={"event":kind,"symbol":symbol,"month":month}
            if error is not None:entry["error_type"]=type(error).__name__
            log.write(json.dumps(entry)+"\n")
            log.flush();os.fsync(log.fileno())
        write("PRECHECK_OK")
        for symbol,month in TARGETS:
            if statuses[(symbol,month)]=="VERIFIED_V23":
                write("ALREADY_VERIFIED_SKIPPED",symbol,month)
                continue
            write("STARTED",symbol,month)
            try:
                ingest(symbol,month,root,v23,cj,cc)
                folder=v23/f"symbol={symbol}"/f"month={month}"
                verify(folder,symbol,month,root,legacy=False)
                write("VERIFIED",symbol,month)
                finished+=1
            except Exception as exc:
                write("FAILED",symbol,month,exc)
                failed={"symbol":symbol,"month":month,"error_type":type(exc).__name__,
                        "message":str(exc)[:300]}
                break
        write("BATCH_FAILED" if failed else "BATCH_DONE")
    after=audit(root,v21,v23)
    summary={"protocol":PROTOCOL,"plan_sha256":_sha(plan_path),
             "newly_verified":finished,
             "total_verified":int(after.status.isin(("VERIFIED_V21","VERIFIED_V23")).sum()),
             "remaining":int(after.status.eq("MISSING").sum()),
             "failed":failed,"journal_sha256":_sha(journal_path),
             "no_pnl_no_orders":True}
    with summary_path.open("x",encoding="utf-8") as f:json.dump(summary,f,indent=2)
    if failed:raise RuntimeError("Ingestion halted: "+str(failed))
    return summary


def main():
    p=argparse.ArgumentParser(description="V25 plan and bounded resume")
    p.add_argument("--mode",required=True,choices=("plan","execute"))
    p.add_argument("--permit-downloads",action="store_true")
    for name in ("data-root","v21-directory","v23-root","coverage-json","coverage-csv",
                 "output-plan","plan-json","journal-jsonl","summary-json"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.mode=="plan":
        if a.permit_downloads: p.error("Plan mode does not permit downloads")
        names=(a.data_root,a.v21_directory,a.v23_root,a.coverage_json,a.coverage_csv,a.output_plan)
        if any(x is None for x in names):p.error("Plan requires all sources and output-plan")
        info=plan(*names)
        print("V25 PLAN",info["targets"],"initial_verified",info["initial_verified"])
        print("NO NETWORK / NO ORDERS")
    else:
        if not all((a.plan_json,a.journal_jsonl,a.summary_json,a.permit_downloads)):
            p.error("Execute requires plan-json, journal-jsonl, summary-json and --permit-downloads")
        summary=execute(a.plan_json,a.journal_jsonl,a.summary_json,a.permit_downloads)
        print("V25",summary["total_verified"],"verified",summary["remaining"],"remaining")
        print("NO PNL / NO ORDERS")


if __name__=="__main__":main()
