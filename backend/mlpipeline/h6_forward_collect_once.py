"""One-shot, fail-closed H6 paper-forecast collector; DOES NOT fetch live data.

Operator supplies externally captured four-symbol final M5 CSV histories.
Runs ONLY for the latest completed shared bar, close to wall-clock time.
Writes immutable per-bar forecast JSON with exclusive-create. No replay,
scheduled tasks, network calls, label reads, order execution or P3 access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlpipeline.features import build_symbol_features, join_market_context
from mlpipeline.h6_forward_candidate_evaluation import (
    START, END, SYMBOLS, VARIANTS, validate_manifest,
)
from mlpipeline.trainer import _class_probs

CANDLE_COLS=("symbol","bar_open_at","bar_close_at","open","high","low","close",
             "volume_base","volume_quote","trade_count","taker_buy_volume_base")
MAX_DELAY=pd.Timedelta(minutes=2)
M5=pd.Timedelta(minutes=5)


def _read_models(bundle:Path):
    manifest_path=bundle/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    models={}
    for variant in VARIANTS:
        spec=manifest["models"][variant]
        location=bundle/f"{variant}.joblib"
        if hashlib.sha256(location.read_bytes()).hexdigest()!=spec["model_sha256"]:
            raise ValueError("Frozen model SHA-256 does not match manifest")
        model=joblib.load(location)  # trusted local artifacts only
        if list(model.feature_name_)!=spec["feature_columns"]:
            raise ValueError("Model columns do not match frozen manifest")
        models[variant]=model
    return manifest,models


def _timestamps(data:pd.Series,label:str):
    # Reject naive/implicit UTC timestamps.
    if not data.astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$",regex=True).all():
        raise ValueError(f"{label}: explicit timezone required")
    values=pd.to_datetime(data,utc=True,errors="raise")
    if values.isna().any():
        raise ValueError(f"Null {label}")
    return values


def prepare_latest_shared_features(candles:pd.DataFrame,*,as_of_utc):
    """Require all four externally supplied finalized 5m bars, same close.

    Exactly one bar gets scored; all features are built using past bars only.
    The CSV is a *snapshot*, never a P0 or sealed-history input directory.
    """
    now=pd.Timestamp(as_of_utc)
    if now.tzinfo is None:
        raise ValueError("UTC-aware clock required")
    now=now.tz_convert("UTC")
    if not set(CANDLE_COLS).issubset(candles.columns) or "is_final" not in candles:
        raise ValueError("Missing OHLCV schema or final marker")
    if candles.empty or set(candles.symbol)!=set(SYMBOLS):
        raise ValueError("Exactly four seen assets required, never ADA")
    if candles.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate bars")
    out=candles.copy()
    out["bar_close_at"]=_timestamps(out["bar_close_at"],"bar_close_at")
    out["bar_open_at"]=_timestamps(out["bar_open_at"],"bar_open_at")
    if (out["is_final"].astype(str).str.lower().isin(("true","1"))==False).any():
        raise ValueError("Only final exchange candles are accepted")
    if not (out.bar_close_at-out.bar_open_at==M5).all():
        raise ValueError("Non-5-minute bar")
    if not ((out.bar_close_at.dt.minute%5==0)&
            (out.bar_close_at.dt.second==0)).all():
        raise ValueError("Unaligned M5 close")
    close=max(out.bar_close_at)
    if not (START<=close<END):
        raise ValueError("Outside prospective frozen window")
    delay=now-close
    if not (pd.Timedelta(0)<=delay<=MAX_DELAY):
        raise ValueError("Too old, unclosed or retrospective candle: no backfill")
    result={}
    for symbol in SYMBOLS:
        s=out.loc[out.symbol.eq(symbol)].sort_values("bar_open_at")
        if s.bar_close_at.iloc[-1]!=close:
            raise ValueError(f"Shared close mismatch: {symbol}")
        if len(s)<100:
            raise ValueError("At least 100 prior M5 bars per symbol needed")
        if s.bar_close_at.gt(close).any():
            raise ValueError("Future bars included")
        result[symbol]=build_symbol_features(s)
    outputs=[]
    for symbol in SYMBOLS:
        merged=join_market_context(result[symbol],result["BTCUSDT"],result["ETHUSDT"])
        row=merged.loc[merged.bar_close_at.eq(close)]
        if len(row)!=1:
            raise ValueError("No exact shared M5 close")
        outputs.append(row)
    return close,outputs


def one_shot(bundle:Path,candle_file:Path,output_dir:Path,*,as_of_utc=None):
    now=pd.Timestamp.now(tz="UTC") if as_of_utc is None else pd.Timestamp(as_of_utc)
    if not now.tzinfo:
        raise ValueError("UTC-aware clock required")
    candle_file=Path(candle_file).resolve()
    output_dir=Path(output_dir).resolve()
    bundle=Path(bundle).resolve()
    if output_dir==bundle or bundle in output_dir.parents:
        raise ValueError("Forecast log must not be inside frozen model directory")
    if candle_file==output_dir or output_dir in candle_file.parents:
        raise ValueError("Input snapshots cannot be inside forecast output directory")
    manifest,models=_read_models(bundle)
    source=pd.read_csv(candle_file)
    close,rows=prepare_latest_shared_features(source,as_of_utc=now)
    # Fail before model prediction if forecast is already recorded.
    dst=output_dir/(close.strftime("%Y%m%dT%H%M%SZ")+".json")
    if dst.exists():
        raise FileExistsError("Immutable forecasts already recorded for M5 close")
    predictions=[]
    for variant in VARIANTS:
        spec=manifest["models"][variant]
        for symbol,row in zip(SYMBOLS,rows):
            X=row.loc[:,spec["feature_columns"]].astype("float32")
            if not np.isfinite(X.to_numpy(dtype=float)).all():
                raise ValueError("Incomplete or nonfinite causal features")
            probs=_class_probs(models[variant],X)[0]
            predictions.append({
                "symbol":symbol,"variant":variant,
                "bar_close_at":close.isoformat(),
                "p_down":float(probs[0]),"p_neutral":float(probs[1]),
                "p_up":float(probs[2]),
            })
    observed=pd.Timestamp.now(tz="UTC")
    if as_of_utc is not None:
        # Injected clock only for tests; real command always uses wall clock.
        observed=now
    if observed>close+MAX_DELAY or observed>=close+pd.Timedelta(minutes=30):
        raise ValueError("Prediction computation missed allowed wall-clock window")
    for row in predictions:
        row["predicted_at_utc"]=observed.isoformat()
    record={
        "protocol":manifest["protocol"],
        "source_snapshot_sha256":hashlib.sha256(candle_file.read_bytes()).hexdigest(),
        "manifest_sha256":hashlib.sha256((bundle/"manifest.json").read_bytes()).hexdigest(),
        "predictions":predictions,
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    with dst.open("x",encoding="utf-8") as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush()
    return dst


def main():
    p=argparse.ArgumentParser(description="One-shot offline H6 paper predictions")
    p.add_argument("--frozen-bundle",required=True,type=Path)
    p.add_argument("--closed-m5-snapshot-csv",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    f=one_shot(a.frozen_bundle,a.closed_m5_snapshot_csv,a.output_dir)
    print("Wrote immutable one-bar paper forecast:",f,flush=True)


if __name__=="__main__":
    main()
