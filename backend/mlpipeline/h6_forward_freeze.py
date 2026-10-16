"""Freeze independent historical-dev H6 30/25 models before forward START.

Standalone research ONLY; nothing is written into P0 or P3. Fold geometry is
derived from timestamps, not held-out outcomes. Features and H6 labels are
constructed ONLY on data strictly before the last development fold's check.
The last eligible training / early-stop / tuning split is predeclared; no
selection based on check, fold-OOS, final90 or ADA outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from mlpipeline.h6_forward_candidate_evaluation import (
    PROTOCOL, START, VARIANTS, columns_for_models, validate_manifest,
)
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

H12 = pd.Timedelta(minutes=60)


def predeclared_last_fold(candles:dict[str,pd.DataFrame]):
    """Choose latest 120/7/3 split using ONLY existing M5 timestamp geometry.

    Last 90 days and fold OOS are excluded downstream BEFORE building labels.
    The generator's synthetic ADA row is metadata only, never market data.
    """
    if set(candles) != set(SEEN_SYMBOLS):
        raise ValueError("Exactly four seen symbols required; never ADA")
    geometry = pd.concat([
        pd.DataFrame({
            "symbol":symbol,
            "bar_close_at":candles[symbol]["bar_close_at"],
            "label_mature_at":candles[symbol]["bar_close_at"]+H12,
        }) for symbol in SEEN_SYMBOLS
    ],ignore_index=True)
    folds=common_folds(geometry)
    fold=folds[-1]
    stop=fold.validation_start+pd.Timedelta(days=7)
    tuning_end=stop+pd.Timedelta(days=3)
    if tuning_end>=fold.test_start:
        raise ValueError("Tune would overlap held-out development check")
    latest=geometry.bar_close_at.max()
    sealed_start=latest+pd.Timedelta(minutes=5)-pd.Timedelta(days=90)
    if fold.test_end>sealed_start or tuning_end>=sealed_start:
        raise ValueError("Freeze boundary intersects reserved final90")
    return fold,stop,tuning_end,sealed_start


def eligible_precheck_examples(candles, tuning_end):
    """Build target/features ONLY from earlier history; no held-out outcomes."""
    clipped={
        symbol:frame.loc[frame.bar_close_at.lt(tuning_end)].copy()
        for symbol,frame in candles.items()
    }
    if any(len(v)<100 for v in clipped.values()):
        raise ValueError("Not enough unsealed history")
    # Building H12 labels on truncated candles drops the last hour, rather
    # than accessing forbidden later closes to complete any future label.
    shared=build_shared_examples(clipped)
    if not shared.label_mature_at.lt(tuning_end).all():
        raise ValueError("Label maturity crossed training/tuning cutoff")
    return shared


def make_frozen_models(candles, *,n_estimators=150,seed=42,now=None):
    if n_estimators!=150 or seed!=42:
        raise ValueError("Frozen preregistration requires 150 estimators and seed 42")
    frozen_at=pd.Timestamp(datetime.now(timezone.utc) if now is None else now)
    if frozen_at.tzinfo is None or frozen_at.tz_convert("UTC")>=START:
        raise ValueError("Freezing must happen before 2026-10-17T00:00:00Z")
    fold,stop,tuning_end,sealed_start=predeclared_last_fold(candles)
    shared=eligible_precheck_examples(candles,tuning_end)
    partitions,bounds=shared_partitions(shared,fold)
    t,mature=shared.bar_close_at,shared.label_mature_at
    tuning=shared.loc[t.ge(stop)&t.lt(tuning_end)&mature.lt(tuning_end)]
    if len(tuning)<100:
        raise ValueError("Too few tuning examples")
    for rows in (*partitions.values(),tuning):
        if set(rows.symbol)!=set(SEEN_SYMBOLS):
            raise ValueError("Four seen assets required in every partition")
    if partitions["validation_check"].empty:
        # The clipped dataset cannot contain check rows by design.
        pass
    else:
        raise AssertionError("Held-out validation check must never be constructed")
    trained={}
    metadata={}
    for variant,cols in columns_for_models().items():
        train=partitions["train"]
        early=partitions["early_stopping"]
        y_train=train.target_h6.to_numpy(dtype=int)
        y_early=early.target_h6.to_numpy(dtype=int)
        counts=np.bincount(y_train,minlength=3)
        if (counts==0).any() or set(y_early)!={0,1,2}:
            raise ValueError("Missing training/early stop class")
        model=_cpu_model(binary=False,n_estimators=n_estimators,seed=seed)
        model.fit(
            train.loc[:,cols].astype("float32"),y_train,
            eval_set=[(early.loc[:,cols].astype("float32"),y_early)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)],
        )
        probs=_class_probs(model,tuning.loc[:,cols].astype("float32"))
        prediction,margin=decision_and_margin(probs,counts,alpha=1.0)
        threshold=threshold_for_target(
            prediction,margin,target_coverage=0.40)
        if not threshold["feasible"]:
            raise ValueError(
                f"{variant}: 40% tuning coverage infeasible; no automatic alternate fold"
            )
        trained[variant]=model
        metadata[variant]={
            "feature_columns":list(cols),
            "train_class_counts":[int(v) for v in counts],
            "threshold":float(threshold["threshold"]),
            "tuning_feasible":True,
            "tuning_selected_coverage":float(
                threshold["tuning_selected_coverage"]),
            "model_frozen_at_utc":frozen_at.isoformat(),
            "last_training_label_mature_at_utc":
                train.label_mature_at.max().isoformat(),
            "last_tuning_bar_close_at_utc":
                tuning.bar_close_at.max().isoformat(),
            "best_iteration":int(model.best_iteration_),
            "train_rows":len(train),
            "early_stop_rows":len(early),
            "tuning_rows":len(tuning),
            "model_file":f"{variant}.joblib",
        }
    manifest={
        "protocol":PROTOCOL,"horizon":6,"coverage_target":0.40,
        "uses_ada":False,"uses_fold_oos":False,
        "uses_reserved_final90":False,
        "models":metadata,
        "feature_schema":FEATURE_SCHEMA_VERSION,
        "label_schema":LABEL_SCHEMA_VERSION,
        "label_threshold_multiplier":0.5,
        "seed":seed,"n_estimators":n_estimators,"alpha":1.0,
        "fold_selection":"last_geometry_fold_no_check_scoring",
        "source_fold_id":int(fold.fold_id),
        "train_start_utc":fold.train_start.isoformat(),
        "validation_start_utc":fold.validation_start.isoformat(),
        "early_stop_end_exclusive_utc":stop.isoformat(),
        "tuning_end_exclusive_utc":tuning_end.isoformat(),
        "sealed_final90_start_utc":sealed_start.isoformat(),
        "shared_h12_purge":True,
        "limitations":[
            "Models use the latest permitted *development* fold, not recent reserved holdout.",
            "Original P0 files are only read, never written.",
            "Manifest self-reports freeze time; an external immutable archive is needed.",
        ],
    }
    return trained,manifest


def write_frozen_bundle(
    output_dir:Path,models:dict,manifest:dict,*,data_root:Path
) -> Path:
    root=Path(output_dir).resolve()
    if root==Path(data_root).resolve() or Path(data_root).resolve() in root.parents:
        raise ValueError("Freeze output must never be beneath P0 root")
    # Pre-serialize and verify before creating the destination directory.
    files={}
    for variant in VARIANTS:
        memory=io.BytesIO()
        joblib.dump(models[variant],memory)
        payload=memory.getvalue()
        manifest["models"][variant]["model_sha256"]=hashlib.sha256(payload).hexdigest()
        files[f"{variant}.joblib"]=payload
    validate_manifest(manifest)
    root.mkdir(parents=True,exist_ok=False)
    for name,payload in files.items():
        with (root/name).open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    with (root/"manifest.json").open("x",encoding="utf-8") as handle:
        json.dump(manifest,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return root/"manifest.json"


def main():
    p=argparse.ArgumentParser(description="Freeze stand-alone H6 30/25 before START")
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--n-estimators",type=int,default=150)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("This frozen protocol requires 24 monthly partitions")
    if a.output_dir.exists():
        p.error("Refusing to overwrite frozen artifacts")
    if (a.output_dir.resolve()==a.data_root.resolve()
            or a.data_root.resolve() in a.output_dir.resolve().parents):
        p.error("Output directory cannot be located under P0")
    if pd.Timestamp.now(tz="UTC")>=START:
        p.error("Locked freeze deadline has passed; amend START first")
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Read-only P0 asset:",symbol,flush=True)
        candles[symbol],entries=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(entries)
    models,manifest=make_frozen_models(
        candles,n_estimators=a.n_estimators,seed=a.seed)
    manifest["historical_dataset_fingerprint_sha256"]=manifest_fingerprint(manifests)
    manifest["lightgbm_version"]=lgb.__version__
    path=write_frozen_bundle(
        a.output_dir,models,manifest,data_root=a.data_root)
    print("Frozen experimental H6 bundle:",path,flush=True)
    print("Archive entire bundle externally before collecting any forecasts.",flush=True)


if __name__=="__main__":
    main()
