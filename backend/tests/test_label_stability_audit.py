"""Synthetic safety/label-schema tests. Never load the historical P0 store."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.label_stability_audit import (
    label_from_return, load_plan, describe_labels, run_audit,
)
from mlpipeline.feature_reduction_diagnostics import PROTOCOL


def plan():
    base=pd.Timestamp("2024-01-01",tz="UTC")
    folds=[]
    for idx in range(36):
        t=base+pd.Timedelta(days=idx*14)
        folds.append({
            "fold_id":idx,
            "train_start_utc":t.isoformat(),
            "validation_start_utc":(t+pd.Timedelta(days=120)).isoformat(),
            "early_stop_end_exclusive_utc":(t+pd.Timedelta(days=127)).isoformat(),
            "unused_tuning_end_exclusive_utc":(t+pd.Timedelta(days=130)).isoformat(),
            "validation_check_end_exclusive_utc":(t+pd.Timedelta(days=134)).isoformat(),
            "partition_rows":{"validation_check":100},
        })
    return {
        "protocol":PROTOCOL,
        "mode":"individual",
        "research_status":"EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "folds_analyzed":36,
        "ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,
        "final_90_days_scored":False,
        "frozen_P3_models_modified":False,
        "folds":folds,
        "dataset_fingerprint_sha256":"example",
    }


def test_fold_geometry_and_safety_flags():
    p=load_plan(plan())
    assert len(p)==36
    assert p[-1]["check_end"] > p[0]["check_end"]
    bad=plan()
    bad["folds"][0]["validation_check_end_exclusive_utc"]="2024-03-01T00:00:00+00:00"
    with pytest.raises(ValueError):
        load_plan(bad)
    bad=plan()
    bad["fold_OOS_scored"]=True
    with pytest.raises(ValueError,match="safety"):
        load_plan(bad)


def test_threshold_boundary_and_changes_are_not_model_accuracy():
    ret=np.array([-0.08,-0.05,0.,0.05,0.08])
    vol=np.ones(5)*0.1
    expected=np.array([0,1,1,1,2])
    assert np.array_equal(label_from_return(ret,vol,2,0.5),expected)
    df=pd.DataFrame({
        "future_return_h2":ret,
        "volatility_48":vol,
        "target_h2":expected,
    })
    d=describe_labels(df,2)
    assert d["counts"]=={"DOWN":1,"NEUTRAL":3,"UP":1}
    assert d["thresholds"]["0.5"]["changed_vs_0_5_fraction"]==0
    assert d["conditional_majority_direction_fraction_among_known_movers"]==0.5


def test_refuse_fingerprint_and_unknown_assets_before_processing():
    with pytest.raises(ValueError,match="fingerprint"):
        run_audit({},plan(),fingerprint="tampered")
    with pytest.raises(ValueError,match="four seen symbols"):
        from mlpipeline.label_stability_audit import shared_with_future_returns
        shared_with_future_returns({"ADAUSDT":pd.DataFrame()},pd.Timestamp(
            "2024-01-01",tz="UTC"))
