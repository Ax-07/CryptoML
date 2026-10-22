"""V26 operator-approved 1-2 target immutable archive ingestion; no trading."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from mlpipeline.funding_basis_batch_resume_v25 import _paths, audit
from mlpipeline.funding_basis_verified_partition_v23 import ingest, validate_coverage
from mlpipeline.funding_basis_partition_registry_v24 import verify
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS

PROTOCOL="p4_v26_flexible_microbatch_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets(raw):
    if not 1 <= len(raw) <= 2:
        raise ValueError("Require exactly one or two explicit targets")
    result=[]
    for item in raw:
        if item.count(":")!=1:
            raise ValueError("Target must have SYMBOL:YYYY-MM syntax")
        symbol,month=item.split(":")
        if symbol not in SYMBOLS or month not in EXPECTED_MONTHS:
            raise ValueError("Target outside frozen P4 universe")
        pair=(symbol,month)
        if pair in result:
            raise ValueError("Duplicate targets")
        result.append(pair)
    return result


def make_plan(source_paths,requested,output):
    root,v21,v23,cj,cc=_paths(*source_paths)
    output=output.resolve()
    if (output.exists() or not output.parent.is_dir() or
        any(output.is_relative_to(p) for p in (root,v21,v23))):
        raise ValueError("New plan outside source roots required")
    chosen=targets(requested)
    frame=audit(root,v21,v23)
    lineage=validate_coverage(cj,cc)
    statuses={(r.symbol,r.month):r.status for r in frame.itertuples(index=False)}
    if any(statuses[t]!="MISSING" for t in chosen):
        raise ValueError("All selected targets must initially be MISSING")
    plan={
        "protocol":PROTOCOL,
        "sources":{"data_root":str(root),"v21_directory":str(v21),
                   "v23_root":str(v23),"coverage_json":str(cj),"coverage_csv":str(cc)},
        "v22":lineage,
        "initial_verified":int(frame.status.isin(("VERIFIED_V21","VERIFIED_V23")).sum()),
        "initial_missing":int(frame.status.eq("MISSING").sum()),
        "targets":[{"symbol":s,"month":m} for s,m in chosen],
        "max_new":len(chosen),"no_pnl":True}
    with output.open("x",encoding="utf-8") as f:
        json.dump(plan,f,indent=2)
    return plan


def execute(plan_path,expected_sha,journal,summary,permit=False):
    if not permit:raise ValueError("Explicit download permission required")
    plan_path=plan_path.resolve();journal=journal.resolve();summary=summary.resolve()
    if (not plan_path.is_file() or journal==summary or
        any(p.exists() or not p.parent.is_dir() for p in (journal,summary))):
        raise ValueError("Existing plan and TWO NEW outputs required")
    digest=sha(plan_path)
    if expected_sha.lower()!=digest:
        raise ValueError("Operator-approved plan SHA256 mismatch")
    plan=json.loads(plan_path.read_bytes())
    if plan.get("protocol")!=PROTOCOL or not plan.get("no_pnl"):
        raise ValueError("Incorrect plan protocol")
    raw=plan.get("targets")
    if not isinstance(raw,list):raise ValueError("Malformed plan targets")
    chosen=targets([f"{x['symbol']}:{x['month']}" for x in raw])
    if plan.get("max_new")!=len(chosen):
        raise ValueError("Plan budget mismatch")
    src=plan["sources"]
    root,v21,v23,cj,cc=_paths(*(Path(src[k]) for k in (
        "data_root","v21_directory","v23_root","coverage_json","coverage_csv")))
    if any(p.is_relative_to(x) for p in (journal,summary) for x in (root,v21,v23)):
        raise ValueError("Output inside protected source directory")
    if validate_coverage(cj,cc)!=plan["v22"]:
        raise ValueError("V22 provenance changed")
    frame=audit(root,v21,v23)
    lookup={(r.symbol,r.month):r.status for r in frame.itertuples(index=False)}
    if any(lookup[t] not in ("MISSING","VERIFIED_V23") for t in chosen):
        raise ValueError("Unsafe or corrupted target")
    completed=0
    failed=None
    with journal.open("x",encoding="utf-8") as stream:
        def event(label,symbol=None,month=None,error=None):
            line={"event":label,"symbol":symbol,"month":month}
            if error is not None:
                line["error_type"]=type(error).__name__
                line["message"]=str(error)[:300]
            stream.write(json.dumps(line)+"\n")
            stream.flush();os.fsync(stream.fileno())
        event("PRECHECK_OK")
        for symbol,month in chosen:
            if lookup[(symbol,month)]=="VERIFIED_V23":
                event("ALREADY_VERIFIED_SKIPPED",symbol,month)
                continue
            event("STARTED",symbol,month)
            try:
                ingest(symbol,month,root,v23,cj,cc)
                verify(v23/f"symbol={symbol}"/f"month={month}",
                       symbol,month,root,legacy=False)
                event("VERIFIED",symbol,month)
                completed+=1
            except Exception as exc:
                failed={"symbol":symbol,"month":month,
                        "error_type":type(exc).__name__,"message":str(exc)[:300]}
                event("FAILED",symbol,month,exc)
                break
        event("BATCH_FAILED" if failed else "BATCH_DONE")
    # Audit the whole registry even after a caught ingestion failure.
    audit_error=None
    try:
        after=audit(root,v21,v23)
        verified=int(after.status.isin(("VERIFIED_V21","VERIFIED_V23")).sum())
        missing=int(after.status.eq("MISSING").sum())
    except Exception as exc:
        audit_error={"error_type":type(exc).__name__,"message":str(exc)[:300]}
        verified=None;missing=None
    result={"protocol":PROTOCOL,"plan_sha256":digest,
            "journal_sha256":sha(journal),"newly_verified":completed,
            "total_verified":verified,"remaining":missing,
            "failed":failed,"final_audit_error":audit_error,
            "no_pnl_no_orders":True}
    with summary.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2)
    if failed or audit_error:
        raise RuntimeError("Microbatch stopped; inspect journal and summary")
    return result


def main():
    p=argparse.ArgumentParser(description="V26 preapproved immutable one/two target microbatch")
    p.add_argument("--mode",choices=("plan","execute"),required=True)
    p.add_argument("--target",action="append",default=[])
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--plan-sha256")
    for field in ("data-root","v21-directory","v23-root","coverage-json","coverage-csv",
                  "output-plan","plan-json","journal-jsonl","summary-json"):
        p.add_argument("--"+field,type=Path)
    a=p.parse_args()
    if a.mode=="plan":
        if a.permit_downloads or a.plan_sha256:
            p.error("No downloads allowed while planning")
        sources=(a.data_root,a.v21_directory,a.v23_root,a.coverage_json,a.coverage_csv)
        if any(x is None for x in sources) or a.output_plan is None:
            p.error("Plan requires five source paths and an output path")
        out=make_plan(sources,a.target,a.output_plan)
        print("V26 PLAN",out["targets"],"verified",out["initial_verified"],
              "missing",out["initial_missing"])
        print("NO NETWORK / NO ORDERS")
    else:
        if (a.target or not a.permit_downloads or
            not all((a.plan_json,a.plan_sha256,a.journal_jsonl,a.summary_json))):
            p.error("Execute requires --permit-downloads, approved plan SHA, and two outputs")
        result=execute(a.plan_json,a.plan_sha256,a.journal_jsonl,a.summary_json,True)
        print("V26 verified",result["total_verified"],"remaining",result["remaining"])
        print("NO PNL / NO ORDERS")


if __name__=="__main__":main()
