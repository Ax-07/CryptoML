"""72h time-boxed H1 paper observer, fresh public Binance M5 only.

Three frozen P2 H1 multiclass LightGBMs (EMA2, EMA2+close_location, 30).
No trading, P3, ADA, fold OOS or final90; no backfilling forecasts.
Run explicitly on user's computer. Observation uses a foreground loop only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.schema import normalize_klines
from ingestion.sources import BinanceClient
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.directional_rule_benchmark import parts_for_fold, majority_direction
from mlpipeline.ema_horizon_grid import EMA_COLUMNS, build_inputs, fold_examples, ema_features, sign
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS, build_symbol_features, join_market_context
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL="p2_h1_72h_prospective_paper_only_v1"
DURATION_HOURS=72
M5=pd.Timedelta(minutes=5)
MODELS={
    "ema2":list(EMA_COLUMNS),
    "ema2_close_location":[*EMA_COLUMNS,"close_location"],
    "lightgbm_30":list(FEATURE_COLUMNS),
}
SYMBOLS=tuple(SEEN_SYMBOLS)


def utc_now()->pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(root:Path)->dict:
    x=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    if (x.get("protocol")!=PROTOCOL or x.get("duration_hours")!=72
        or tuple(x.get("symbols",[]))!=SYMBOLS
        or x.get("model_columns")!=MODELS):
        raise ValueError("Unexpected observer protocol")
    start,end=pd.Timestamp(x["start_utc"]),pd.Timestamp(x["end_exclusive_utc"])
    if (start.tzinfo is None or end.tzinfo is None or end-start!=pd.Timedelta(hours=72)):
        raise ValueError("Invalid short prospective period")
    return x


def freeze(*,data_root:Path,fold_plan:Path,output:Path,verify_hashes:bool)->dict:
    """Use only last ORIGINAL P2 fold train/early; no check training or scoring."""
    if output.exists() or data_root.resolve() in output.resolve().parents or (
        data_root.resolve().parent in output.resolve().parents):
        raise ValueError("Freeze output exists or is within production tree")
    source=json.loads(fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(source)
    if len(plan)!=36:
        raise ValueError("Full fixed P2 plan required")
    store=CandleStore(data_root)
    candles,manifests={},[]
    for sym in SYMBOLS:
        print("Historical P0 consultation only:",sym,flush=True)
        candles[sym],files=load_symbol(store,sym,months=24,verify_hashes=verify_hashes)
        manifests.extend(files)
    if manifest_fingerprint(manifests)!=source["dataset_fingerprint_sha256"]:
        raise ValueError("Immutable original P0 fingerprint mismatch")
    now=utc_now()
    start=now.floor("5min")+pd.Timedelta(minutes=10)
    latest=max(df.bar_close_at.max() for df in candles.values())
    if latest>=start:
        raise ValueError("Future validation start overlaps original P0 history")
    # Clip feature/label computation strictly to last allowed P2 check.
    fold=plan[-1]
    contexts,ema=build_inputs(candles,fold["check_end"])
    shared=fold_examples(contexts,ema,fold["check_end"])
    parts,_=parts_for_fold(shared,fold,source["folds"][fold["fold_id"]])
    train,early=parts["train"],parts["early_stopping"]
    yt=train.target_h1.to_numpy(dtype=int)
    ye=early.target_h1.to_numpy(dtype=int)
    if (set(yt)!={0,1,2} or set(ye)!={0,1,2}):
        raise ValueError("Missing historical train/early class")
    fitted={}
    iterations={}
    for name,columns in MODELS.items():
        clf=_cpu_model(binary=False,n_estimators=150,seed=42)
        clf.fit(train[columns].astype("float32"),yt,
                eval_set=[(early[columns].astype("float32"),ye)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
        fitted[name]=clf
        iterations[name]=int(clf.best_iteration_)
        print("Frozen candidate",name,"iteration",iterations[name],flush=True)
    output.mkdir(parents=True,exist_ok=False)
    payload={"estimators":fitted,"majority_direction":majority_direction(yt)}
    model_path=output/"models.joblib"
    joblib.dump(payload,model_path,compress=3)
    document={
        "protocol":PROTOCOL,"duration_hours":DURATION_HOURS,
        "created_at_utc":now.isoformat(),
        "start_utc":start.isoformat(),
        "end_exclusive_utc":(start+pd.Timedelta(hours=72)).isoformat(),
        "symbols":list(SYMBOLS),"model_columns":MODELS,
        "seed":42,"n_estimators_max":150,"best_iterations":iterations,
        "original_fold_id":fold["fold_id"],"training_on_original_P2_train_and_early_only":True,
        "historical_last_allowed_check_end_exclusive_utc":fold["check_end"].isoformat(),
        "original_P0_max_close_utc":latest.isoformat(),
        "source_P0_fingerprint_sha256":manifest_fingerprint(manifests),
        "source_fold_plan_sha256":sha(fold_plan),
        "model_file_sha256":sha(model_path),
        "no_ADA_no_fold_OOS_no_final90_no_P3":True,
        "no_retraining_during_72h":True,"fresh_REST_only_after_start_no_backfill":True,
        "note":"First fresh bars are warmup; NO prediction scored retrospectively.",
    }
    with (output/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(document,f,indent=2,ensure_ascii=False)
    print("Frozen 72h H1 test:",output,flush=True)
    print("UTC start",document["start_utc"],"UTC end",document["end_exclusive_utc"],flush=True)
    return document


def valid_capture_clock(meta:dict,now:pd.Timestamp,t:pd.Timestamp)->bool:
    start=pd.Timestamp(meta["start_utc"])
    end=pd.Timestamp(meta["end_exclusive_utc"])
    return bool(start<=t<end and now>=t+pd.Timedelta(seconds=5)
                and now<t+M5-pd.Timedelta(seconds=10) and now<end)


def collect_once(root:Path,*,client:BinanceClient,now:pd.Timestamp|None=None)->str:
    """Predict ONLY latest already-closed M5, while its future close is unknown."""
    meta=_meta(root)
    now=utc_now() if now is None else now
    start,end=pd.Timestamp(meta["start_utc"]),pd.Timestamp(meta["end_exclusive_utc"])
    if now<start+pd.Timedelta(hours=4,minutes=15):
        return "Warmup: waiting for 49+ fresh M5 bars"
    if now>=end:
        return "Observation window ended; no new predictions"
    model_path=root/"models.joblib"
    if sha(model_path)!=meta["model_file_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    # Trusted locally trained model file ONLY; never accept external joblib.
    packed=joblib.load(model_path)
    rows={}
    as_of=now-pd.Timedelta(seconds=3)
    for sym in SYMBOLS:
        raw=client.fetch_rest(symbol=sym,start=start.to_pydatetime(),
                              end=now.to_pydatetime(),as_of=as_of.to_pydatetime())
        bars=normalize_klines(raw,symbol=sym,source="fresh_short72_public_rest",
                               as_of=as_of.to_pydatetime())
        if bars.empty or bars.bar_open_at.min()!=start:
            return "Missing initial fresh bars for "+sym+"; skip, never backfill"
        rows[sym]=bars
    last={s:df.bar_close_at.max() for s,df in rows.items()}
    if len(set(last.values()))!=1:
        return "Four assets have nonmatching last close; skip"
    t=next(iter(last.values()))
    if not valid_capture_clock(meta,utc_now(),t):
        return "Latest 5m bar stale/not mature or outside 72h; skip"
    if any(not df.bar_open_at.diff().dropna().eq(M5).all() for df in rows.values()):
        return "Fresh data gap; skip until complete contiguous context"
    candle_features={s:build_symbol_features(rows[s]) for s in SYMBOLS}
    frames={}
    for sym in SYMBOLS:
        common=join_market_context(
            candle_features[sym],candle_features["BTCUSDT"],
            candle_features["ETHUSDT"])
        ea=ema_features(rows[sym])
        merged=common.merge(ea,on=["symbol","bar_close_at"],
                             how="left",sort=False,validate="one_to_one")
        x=merged.loc[merged.bar_close_at.eq(t)]
        if len(x)!=1 or not np.isfinite(x[list(FEATURE_COLUMNS)+list(EMA_COLUMNS)].to_numpy(dtype=float)).all():
            return "Features not mature/finite on all four assets; skip"
        frames[sym]=x
    path=root/"predictions.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item=json.loads(line)
            if item["bar_close_at_utc"]==t.isoformat():
                return "Already predicted this bar; no overwrite"
    packet={"bar_close_at_utc":t.isoformat(),"matures_at_utc":(t+M5).isoformat(),
            "predictions":[]}
    for sym in SYMBOLS:
        x=frames[sym]
        one={"symbol":sym,"close_at_t":float(x.close.iloc[0])}
        for name,columns in MODELS.items():
            p=_class_probs(packed["estimators"][name],x[columns].astype("float32"))
            one[name]={"sign":int(sign(p,packed["majority_direction"])[0]),
                       "probabilities_D_N_U":p[0].tolist()}
        # Reference uses PREVIOUS closed M5 return only.
        one["previous_reversion_sign"]=int(2 if x.ret_1.iloc[0]<0 else 0)
        packet["predictions"].append(one)
    # Check again AFTER all network and model inference; never append after target matures.
    write_time=utc_now()
    if not valid_capture_clock(meta,write_time,t):
        return "Prediction computed too late; reject, never backfill"
    packet["committed_at_utc"]=write_time.isoformat()
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(packet,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())
    return f"Recorded causally at {write_time.isoformat()}: {t.isoformat()} x 4 assets"


def follow(root:Path)->None:
    """Foreground only, bounded; CTRL+C safe. No service, no daemon."""
    meta=_meta(root)
    lock=root/".observer.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode())
        with BinanceClient() as client:
            while utc_now()<pd.Timestamp(meta["end_exclusive_utc"]):
                now=utc_now()
                next_poll=now.floor("5min")+pd.Timedelta(seconds=14)
                if next_poll<=now:
                    next_poll+=M5
                remaining=(next_poll-now).total_seconds()
                time.sleep(max(0,min(remaining,300)))
                if utc_now()>=pd.Timestamp(meta["end_exclusive_utc"]):
                    break
                try:
                    print(collect_once(root,client=client),flush=True)
                except Exception as exc:
                    print("Skipped latest close due to:",repr(exc),flush=True)
        print("72h observation window has ended",flush=True)
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def score(root:Path,*,client:BinanceClient,now:pd.Timestamp|None=None)->dict:
    meta=_meta(root)
    now=utc_now() if now is None else now
    start,end=pd.Timestamp(meta["start_utc"]),pd.Timestamp(meta["end_exclusive_utc"])
    if now<end+pd.Timedelta(minutes=6):
        raise ValueError("Do not inspect scores before the entire window is mature")
    if (root/"final_report.json").exists():
        raise ValueError("Final report already exists; refuse rewrite")
    if (root/".observer.lock").exists():
        raise ValueError("Observer still running/lock remains")
    journal=root/"predictions.jsonl"
    if not journal.exists():
        raise ValueError("No genuinely recorded forecasts")
    packets=[json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines()]
    if not packets:
        raise ValueError("Empty prediction ledger")
    observed={}
    for item in packets:
        t=pd.Timestamp(item["bar_close_at_utc"])
        issued=pd.Timestamp(item["committed_at_utc"])
        if (t in observed or issued>=t+M5 or issued<=t
            or not start<=t<end or len(item["predictions"])!=4
            or {p["symbol"] for p in item["predictions"]}!=set(SYMBOLS)):
            raise ValueError("Duplicate/late/invalid forecast packet")
        observed[t]=item
    truth={}
    for sym in SYMBOLS:
        raw=client.fetch_rest(symbol=sym,start=start.to_pydatetime(),
                              end=(end+M5).to_pydatetime(),as_of=now.to_pydatetime())
        bars=normalize_klines(raw,symbol=sym,
            source="fresh_short72_public_rest",as_of=now.to_pydatetime())
        if bars.empty or bars.bar_open_at.min()!=start:
            raise ValueError("Missing new-period market data for "+sym)
        price=bars.set_index("bar_close_at")["close"]
        truth[sym]=price
    results=[]
    for t,item in sorted(observed.items()):
        for entry in item["predictions"]:
            sym=entry["symbol"]
            closes=truth[sym]
            if t not in closes.index or t+M5 not in closes.index:
                raise ValueError("Unconfirmed t/t+5 actual candle; no fabricated truth")
            close_t=float(closes.loc[t])
            close_next=float(closes.loc[t+M5])
            if not np.isclose(close_t,entry["close_at_t"],rtol=1e-11,atol=1e-9):
                raise ValueError("Exchange history differs from prediction-time close")
            actual=int(2 if close_next>close_t else 0 if close_next<close_t else 1)
            result={"symbol":sym,"t_utc":t.isoformat(),"actual":actual}
            for name in MODELS:
                result[name]=entry[name]["sign"]
            result["always_down"]=0
            result["previous_reversion"]=entry["previous_reversion_sign"]
            results.append(result)
    frame=pd.DataFrame(results)
    flat=int(frame.actual.eq(1).sum())
    valid=frame.loc[frame.actual.ne(1)]
    if valid.empty:
        raise ValueError("No nonflat future bars observed")
    names=[*MODELS,"always_down","previous_reversion"]
    def metrics(subset):
        if subset.empty:
            return None
        return {"n":len(subset),
                "accuracy":{name:float((subset[name]==subset.actual).mean())
                            for name in names}}
    total=metrics(valid)
    paired={}
    for name in MODELS:
        for baseline in ("always_down","previous_reversion","ema2"):
            if baseline==name:
                continue
            a=valid[name].eq(valid.actual)
            b=valid[baseline].eq(valid.actual)
            paired[f"{name}_minus_{baseline}"]={
                "delta_points":100*float(a.mean()-b.mean()),
                "unique_model_correct":int((a&~b).sum()),
                "unique_reference_correct":int((~a&b).sum()),
            }
    return {"protocol":PROTOCOL,"status":"SHORT72_PAPER_PROSPECTIVE_EXPLORATORY",
            "window_start_utc":meta["start_utc"],
            "window_end_utc":meta["end_exclusive_utc"],
            "prediction_packets":len(packets),"evaluated_nonflat_rows":len(valid),
            "excluded_flat_rows":flat,"pooled":total,
            "by_symbol":{sym:metrics(valid.loc[valid.symbol.eq(sym)]) for sym in SYMBOLS},
            "paired":paired,
            "limitations":["72 hours only; not a long-term or independent statistical proof",
                           "Same timestamps and correlated crypto assets",
                           "No trades/costs/slippage; no automatic selection or promotion",
                           "Dropped/gapped predictions are never backfilled"],
            "no_P3_ADA_OOS_or_final90":True}


def main()->None:
    p=argparse.ArgumentParser(description="72h short prospective H1 paper-only validation")
    sub=p.add_subparsers(dest="action",required=True)
    freeze_p=sub.add_parser("freeze")
    freeze_p.add_argument("--data-root",required=True,type=Path)
    freeze_p.add_argument("--fold-plan",required=True,type=Path)
    freeze_p.add_argument("--output-dir",required=True,type=Path)
    freeze_p.add_argument("--verify-hashes",action="store_true")
    observe_p=sub.add_parser("observe")
    observe_p.add_argument("--experiment-dir",required=True,type=Path)
    observe_p.add_argument("--follow",action="store_true")
    eval_p=sub.add_parser("evaluate")
    eval_p.add_argument("--experiment-dir",required=True,type=Path)
    a=p.parse_args()
    if a.action=="freeze":
        freeze(data_root=a.data_root,fold_plan=a.fold_plan,
               output=a.output_dir,verify_hashes=a.verify_hashes)
    elif a.action=="observe":
        if a.follow:
            follow(a.experiment_dir)
        else:
            with BinanceClient() as client:
                print(collect_once(a.experiment_dir,client=client),flush=True)
    else:
        with BinanceClient() as client:
            result=score(a.experiment_dir,client=client)
        with (a.experiment_dir/"final_report.json").open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        print("Short 72h paper report:",a.experiment_dir/"final_report.json",flush=True)
        print("H1 accuracy:",result["pooled"]["accuracy"],flush=True)


if __name__=="__main__":
    main()
