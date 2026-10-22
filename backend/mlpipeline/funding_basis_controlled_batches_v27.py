"""V27 approved 1-4 immutable source partitions per batch; no trading."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from mlpipeline.funding_basis_batch_resume_v25 import _paths,audit
from mlpipeline.funding_basis_verified_partition_v23 import ingest,validate_coverage
from mlpipeline.funding_basis_partition_registry_v24 import verify
from mlpipeline.pullback_long_history_v3 import SYMBOLS,EXPECTED_MONTHS

PROTOCOL="p4_v27_operator_approved_bounded_batches_v1"
MIN_FREE=512*1024*1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_targets(items):
    if not 1<=len(items)<=4:raise ValueError("Require 1 to 4 explicit targets")
    output=[]
    for value in items:
        parts=value.split(":")
        if len(parts)!=2 or parts[0] not in SYMBOLS or parts[1] not in EXPECTED_MONTHS:
            raise ValueError("Target outside frozen universe or wrong format")
        entry=(parts[0],parts[1])
        if entry in output:raise ValueError("Duplicate target")
        if entry==("BTCUSDT","2025-06"):raise ValueError("Legacy V21 target cannot be reingested")
        output.append(entry)
    return output


def snapshot(frame,v21,v23):
    """Bind every existing verified partition to its on-disk manifest digest."""
    found={}
    for row in frame.itertuples(index=False):
        if row.status in ("VERIFIED_V21","VERIFIED_V23"):
            p=(v21 if row.status=="VERIFIED_V21" else
               v23/f"symbol={row.symbol}"/f"month={row.month}")
            found[f"{row.symbol}:{row.month}"]=sha(p/"manifest.json")
    return found


def plan(sources,items,output):
    root,v21,v23,cj,cc=_paths(*sources)
    output=output.resolve()
    if (output.exists() or not output.parent.is_dir() or
        any(output.is_relative_to(x) for x in (root,v21,v23))):
        raise ValueError("Plan output must be NEW outside protected datasets")
    chosen=parse_targets(items)
    frame=audit(root,v21,v23)
    lookup={(r.symbol,r.month):r.status for r in frame.itertuples(index=False)}
    if any(lookup[t]!="MISSING" for t in chosen):
        raise ValueError("All requested partitions must initially be MISSING")
    report={"protocol":PROTOCOL,
            "sources":dict(zip(("data_root","v21_directory","v23_root","coverage_json","coverage_csv"),
                               (str(root),str(v21),str(v23),str(cj),str(cc)))),
            "v22":validate_coverage(cj,cc),
            "verified_manifests":snapshot(frame,v21,v23),
            "initial_verified":len(snapshot(frame,v21,v23)),
            "initial_missing":int(frame.status.eq("MISSING").sum()),
            "targets":[{"symbol":s,"month":m} for s,m in chosen],
            "max_new":len(chosen),"min_free_bytes":MIN_FREE,"no_pnl_no_orders":True}
    with output.open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    return report


def execute(plan_path,approved_sha,journal,summary,allow=False):
    if not allow:raise ValueError("Explicit --permit-downloads required")
    plan_path=plan_path.resolve();journal=journal.resolve();summary=summary.resolve()
    if (not plan_path.is_file() or journal==summary or
        any(x.exists() or not x.parent.is_dir() for x in (journal,summary))):
        raise ValueError("Plan required and journal/summary must be NEW")
    digest=sha(plan_path)
    if digest!=approved_sha.lower():raise ValueError("Operator SHA256 approval mismatch")
    data=json.loads(plan_path.read_bytes())
    if data.get("protocol")!=PROTOCOL or data.get("no_pnl_no_orders") is not True:
        raise ValueError("Invalid plan protocol")
    raw=data.get("targets")
    if not isinstance(raw,list):raise ValueError("Invalid targets")
    try:
        chosen=parse_targets([r["symbol"]+":"+r["month"] for r in raw])
    except (TypeError,KeyError) as exc:raise ValueError("Malformed targets") from exc
    if data.get("max_new")!=len(chosen) or data.get("min_free_bytes")!=MIN_FREE:
        raise ValueError("Invalid budget")
    try:
        s=data["sources"]
        root,v21,v23,cj,cc=_paths(*(Path(s[k]) for k in
            ("data_root","v21_directory","v23_root","coverage_json","coverage_csv")))
    except (TypeError,KeyError) as exc:raise ValueError("Malformed source paths") from exc
    if any(p.is_relative_to(x) for p in (journal,summary) for x in (root,v21,v23)):
        raise ValueError("Output inside protected datasets")
    if validate_coverage(cj,cc)!=data["v22"]:
        raise ValueError("Changed V22 lineage")
    before=audit(root,v21,v23)
    previous=snapshot(before,v21,v23)
    if any(previous.get(k)!=v for k,v in data["verified_manifests"].items()):
        raise ValueError("Previously verified manifest changed or disappeared")
    statuses={(r.symbol,r.month):r.status for r in before.itertuples(index=False)}
    if any(statuses[t] not in ("MISSING","VERIFIED_V23") for t in chosen):
        raise ValueError("Unsafe target state")
    completed=0;failure=None
    with journal.open("x",encoding="utf-8") as out:
        def emit(label,symbol=None,month=None,detail=None):
            record={"event":label,"symbol":symbol,"month":month}
            if detail is not None:record["detail"]=detail
            out.write(json.dumps(record)+"\n");out.flush();os.fsync(out.fileno())
        emit("PRECHECK_OK")
        for symbol,month in chosen:
            if statuses[(symbol,month)]=="VERIFIED_V23":
                emit("ALREADY_VERIFIED_SKIPPED",symbol,month)
                continue
            try:
                free=shutil.disk_usage(v23).free
                if free<MIN_FREE:raise ValueError("Insufficient free disk space")
                emit("STARTED",symbol,month)
                ingest(symbol,month,root,v23,cj,cc)
                folder=v23/f"symbol={symbol}"/f"month={month}"
                verify(folder,symbol,month,root,legacy=False)
                emit("VERIFIED",symbol,month,sha(folder/"manifest.json"))
                completed+=1
            except Exception as exc:
                failure={"symbol":symbol,"month":month,"error_type":type(exc).__name__,
                         "message":str(exc)[:300]}
                emit("FAILED",symbol,month,failure)
                break
        emit("BATCH_FAILED" if failure else "BATCH_DONE")
    audit_error=None
    try:
        after=audit(root,v21,v23)
        total=int(after.status.isin(("VERIFIED_V21","VERIFIED_V23")).sum())
        missing=int(after.status.eq("MISSING").sum())
    except Exception as exc:
        total=None;missing=None
        audit_error={"error_type":type(exc).__name__,"message":str(exc)[:300]}
    result={"protocol":PROTOCOL,"plan_sha256":digest,"journal_sha256":sha(journal),
            "newly_verified":completed,"total_verified":total,"remaining":missing,
            "failed":failure,"final_audit_error":audit_error,"no_pnl_no_orders":True}
    with summary.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    if failure or audit_error:raise RuntimeError("Batch stopped; inspect saved journal and summary")
    return result


def main():
    p=argparse.ArgumentParser(description="V27 bounded, operator-approved 1-4 partition intake")
    p.add_argument("--mode",choices=("plan","execute"),required=True)
    p.add_argument("--target",action="append",default=[])
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--plan-sha256")
    for key in ("data-root","v21-directory","v23-root","coverage-json","coverage-csv",
                "output-plan","plan-json","journal-jsonl","summary-json"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.mode=="plan":
        paths=(a.data_root,a.v21_directory,a.v23_root,a.coverage_json,a.coverage_csv)
        if a.permit_downloads or a.plan_sha256 or a.output_plan is None or any(x is None for x in paths):
            p.error("Plan is offline and requires 5 source paths + output-plan")
        report=plan(paths,a.target,a.output_plan)
        print("V27 PLAN",report["targets"],"verified",report["initial_verified"],
              "missing",report["initial_missing"])
        print("NO NETWORK / NO ORDERS")
    else:
        if (a.target or not a.permit_downloads or
            not all((a.plan_json,a.plan_sha256,a.journal_jsonl,a.summary_json))):
            p.error("Execute needs approved plan hash, journal, summary and explicit permission")
        result=execute(a.plan_json,a.plan_sha256,a.journal_jsonl,a.summary_json,True)
        print("V27",result["total_verified"],"verified",result["remaining"],"missing")
        print("NO PNL / NO ORDERS")

if __name__=="__main__":main()
