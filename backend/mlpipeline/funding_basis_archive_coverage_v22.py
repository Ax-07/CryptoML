"""V22 opt-in HEAD-only archive coverage audit. NO ZIP downloads/trades."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pandas as pd

from mlpipeline.funding_basis_archive_integrity_v19 import BASE, PATHS
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS

PROTOCOL="p4_v22_public_archive_head_coverage_v1"
SAMPLE_MONTHS=("2024-10","2025-06","2026-06")
KINDS=("spot","perp","mark","funding")
FIELDS=("symbol","month","kind","status","http","content_length","last_modified","error_type")


def inventory(mode:str):
    if mode not in ("sample","full"):
        raise ValueError("Mode must be sample or full")
    months=SAMPLE_MONTHS if mode=="sample" else EXPECTED_MONTHS
    records=[]
    for symbol in SYMBOLS:
        for month in months:
            for kind in KINDS:
                url=BASE+PATHS[kind].format(s=symbol,m=month)
                records.append((symbol,month,kind,url))
    assert len(records)==len(SYMBOLS)*len(months)*len(KINDS)
    return records


def head(url:str)->dict:
    try:
        with urlopen(Request(url,method="HEAD",
                             headers={"User-Agent":"CryptoML-V22-coverage-audit"}),timeout=12) as response:
            status=int(response.status)
            return {"http":status,
                    "status":"HEAD_OK_UNVERIFIED" if status==200 else "UNVERIFIED",
                    "content_length":response.headers.get("Content-Length"),
                    "last_modified":response.headers.get("Last-Modified"),
                    "error_type":None}
    except HTTPError as exc:
        return {"http":exc.code,
                "status":"HEAD_404_NOT_CONFIRMED" if exc.code==404 else "UNVERIFIED",
                "content_length":None,"last_modified":None,"error_type":None}
    except (OSError,URLError,TimeoutError) as exc:
        return {"http":None,"status":"UNVERIFIED","content_length":None,
                "last_modified":None,"error_type":type(exc).__name__}


def one(item):
    symbol,month,kind,url=item
    return {"symbol":symbol,"month":month,"kind":kind,**head(url)}


def summarize(rows:list[dict],mode:str):
    expected=inventory(mode)
    if len(rows)!=len(expected):
        raise ValueError("Coverage incomplete")
    actual={(r["symbol"],r["month"],r["kind"]) for r in rows}
    if actual!={(s,m,k) for s,m,k,_ in expected}:
        raise ValueError("Coverage keys inconsistent")
    output=pd.DataFrame(rows,columns=FIELDS)
    if not output.status.isin(("HEAD_OK_UNVERIFIED","HEAD_404_NOT_CONFIRMED","UNVERIFIED")).all():
        raise ValueError("Unknown status")
    def group(cols):
        return [{"keys":list(x if isinstance(x,tuple) else (x,)),
                 "total":len(v),"head_200":int(v.status.eq("HEAD_OK_UNVERIFIED").sum()),
                 "not_confirmed":int(v.status.ne("HEAD_OK_UNVERIFIED").sum())}
                for x,v in output.groupby(cols,sort=True)]
    report={"protocol":PROTOCOL,"mode":mode,"universes":list(SYMBOLS),
            "months":list(SAMPLE_MONTHS if mode=="sample" else EXPECTED_MONTHS),
            "kinds":list(KINDS),"expected":len(expected),"observed":len(output),
            "head_200":int(output.status.eq("HEAD_OK_UNVERIFIED").sum()),
            "head_404":int(output.status.eq("HEAD_404_NOT_CONFIRMED").sum()),
            "unverified":int(output.status.eq("UNVERIFIED").sum()),
            "by_symbol":group(["symbol"]),"by_kind":group(["kind"]),
            "by_month":group(["month"]),
            "no_downloads":True,"no_data_integrity_verified":True,
            "no_backtest_or_trading":True,
            "limitations":"HEAD 200 does not verify ZIP/checksum/schema or completeness of the trading histories."}
    return report,output


def main():
    p=argparse.ArgumentParser(description="V22 HEAD inventory: sample=48, full=336, no ZIP download")
    p.add_argument("--probe-public-archives",action="store_true")
    p.add_argument("--mode",choices=("sample","full"),default="sample")
    p.add_argument("--output-json",required=True,type=Path)
    p.add_argument("--output-csv",required=True,type=Path)
    a=p.parse_args()
    if not a.probe_public_archives:
        p.error("Explicit --probe-public-archives required")
    paths=[a.output_json.resolve(),a.output_csv.resolve()]
    if paths[0]==paths[1] or any(x.exists() or not x.parent.is_dir() for x in paths):
        p.error("Require two NEW distinct output paths")
    jobs=inventory(a.mode)
    print("V22",a.mode,"HEAD requests",len(jobs),flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows=list(executor.map(one,jobs))
    report,frame=summarize(rows,a.mode)
    with paths[1].open("x",encoding="utf-8",newline="") as f:
        frame.to_csv(f,index=False)
    with paths[0].open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    print("HEAD 200",report["head_200"],"/",report["expected"],
          "404",report["head_404"],"other",report["unverified"],flush=True)
    print("NO ZIP DOWNLOADS / NO INTEGRITY CLAIM / NO ORDERS")


if __name__=="__main__":main()
