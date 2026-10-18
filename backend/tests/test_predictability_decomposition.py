"""No historical P0 inputs: synthetic probability, regime and safety tests."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.predictability_decomposition import (
    _binary_scores, _prior_from_train, stage_scores,
    training_regime_thresholds, regime_codes, run_audit,
)
from mlpipeline.feature_reduction_diagnostics import PROTOCOL as SOURCE_PROTOCOL


def test_movement_and_direction_are_distinct():
    y=np.array([0,1,2,1,0,2])
    probs=np.array([
        [0.85,0.10,0.05], [0.10,0.80,0.10], [0.05,0.10,0.85],
        [0.15,0.70,0.15], [0.10,0.10,0.80], [0.80,0.10,0.10],
    ])
    result=stage_scores(
        y,probs,training_movement_prior=0.6,
        training_up_given_movement=0.5)
    a=result["movement_detection"]
    b=result["direction_given_TRUE_movement_ORACLE"]
    assert a["n"]==6 and a["positive"]==4
    assert a["average_precision"] > a["observed_positive_fraction"]
    assert b["n"]==4 and b["positive"]==2
    assert b["accuracy_oracle_true_movers"]==pytest.approx(0.5)
    assert b["train_majority_accuracy_on_true_movers"]==pytest.approx(0.5)


def test_train_prior_is_used_for_probability_scoring():
    y=np.array([1,0,0,1,1,1])
    r=_binary_scores(y,np.repeat(0.7,len(y)),train_prior=0.3)
    assert r["train_positive_prior"]==pytest.approx(0.3)
    assert r["observed_positive_fraction"]==pytest.approx(4/6)
    assert r["train_prior_brier"]!=pytest.approx(
        np.mean((y-y.mean())**2))
    assert r["brier_skill_vs_train_prior"]!=0


def test_no_oracle_gate_and_invalid_probabilities():
    with pytest.raises(ValueError):
        stage_scores(
            np.array([1,2]), np.array([[0.1,0.9,0.0],[0.1,0.4,0.4]]),
            training_movement_prior=0.6,
            training_up_given_movement=0.5)
    assert _prior_from_train(np.array([0,1,2,1]))==(0.5,0.5)
    with pytest.raises(ValueError):
        _prior_from_train(np.array([1,1,1]))


def test_regimes_use_training_quantiles_per_symbol():
    rng=np.random.default_rng(2)
    names=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
    frame=pd.concat([
        pd.DataFrame({"symbol":symbol,
                      "volatility_48":np.linspace(0.001+idx,0.9+idx,120)})
        for idx,symbol in enumerate(names)
    ],ignore_index=True)
    splits=training_regime_thresholds(frame)
    test=frame.iloc[[0,60,119,120,180,239]].copy()
    codes=regime_codes(test,splits)
    assert codes.tolist()==[0,1,2,0,1,2]
    with pytest.raises(ValueError,match="Unknown asset"):
        regime_codes(pd.DataFrame({"symbol":["ADAUSDT"],"volatility_48":[1.]}),splits)


def test_no_p0_access_if_report_fingerprint_mismatch():
    # run_audit rejects before constructing labels from candles
    base=pd.Timestamp("2024-01-01",tz="UTC")
    folds=[]
    for idx in range(36):
        t=base+pd.Timedelta(days=14*idx)
        folds.append({
            "fold_id":idx,
            "train_start_utc":t.isoformat(),
            "validation_start_utc":(t+pd.Timedelta(days=120)).isoformat(),
            "early_stop_end_exclusive_utc":(t+pd.Timedelta(days=127)).isoformat(),
            "unused_tuning_end_exclusive_utc":(t+pd.Timedelta(days=130)).isoformat(),
            "validation_check_end_exclusive_utc":(t+pd.Timedelta(days=134)).isoformat(),
            "partition_rows":{"validation_check":100,"train":100,"early_stopping":100}
        })
    src={
        "protocol":SOURCE_PROTOCOL,"mode":"individual",
        "research_status":"EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "folds_analyzed":36,"folds":folds,
        "dataset_fingerprint_sha256":"source_hash",
        "ADA_loaded_or_scored":False,"fold_OOS_scored":False,
        "final_90_days_scored":False,"frozen_P3_models_modified":False}
    with pytest.raises(ValueError,match="fingerprint"):
        run_audit({},src,fingerprint="other_hash",max_folds=1)
