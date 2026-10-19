"""Synthetic-only tests for predeclared 30/32 trajectory paired experiment."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.trajectory_direction_comparison import (
    align_features, direction_sign, score_pair, verify_original_fold,
    summarize, VARIANTS, BASELINE, CANDIDATE,
)
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.trajectory_direction_features import TRAJECTORY_COLUMNS


def test_variant_schema_unchanged_and_join_alignment():
    assert tuple(VARIANTS[BASELINE])==tuple(FEATURE_COLUMNS)
    assert len(VARIANTS[CANDIDATE])==32
    dt=pd.date_range("2025-01-01",periods=3,freq="5min",tz="UTC")
    base=pd.DataFrame({"symbol":["BTCUSDT"]*3,"bar_close_at":dt,
                       "target_h2":[0,1,2]})
    extra=pd.DataFrame({
        "symbol":["BTCUSDT"]*3,"bar_close_at":dt[::-1],
        TRAJECTORY_COLUMNS[0]:[1.,2.,3.],
        TRAJECTORY_COLUMNS[1]:[0.,1.,2.],
    })
    observed=align_features(base,extra)
    assert observed["target_h2"].tolist()==[0,1,2]
    assert observed["bar_close_at"].tolist()==base["bar_close_at"].tolist()
    assert observed[TRAJECTORY_COLUMNS[0]].tolist()==[3.,2.,1.]
    missing=extra.iloc[:2].copy()
    with pytest.raises(ValueError,match="warmup"):
        align_features(base,missing)


def test_direction_forced_binary_and_neutral_wrong():
    ytrain=np.array([0,0,1,2,2])
    probs_b=np.array([[0.7,0.1,0.2],[0.2,0.7,0.1],[0.5,0.,0.5]])
    probs_c=np.array([[0.1,0.4,0.5],[0.1,0.3,0.6],[0.5,0.,0.5]])
    assert direction_sign(probs_b,ytrain).tolist()==[0,0,0]
    result=score_pair(
        np.array([2,1,0]),probs_b,probs_c,
        np.array([True,True,True]),ytrain,np.array([0,0,2]))
    assert result["n"]==3
    assert result["baseline"]["correct"]==1
    assert result["candidate"]["correct"]==2
    assert result["baseline"]["neutral_errors"]==1
    assert result["candidate_vs_baseline_disagreements"]["n"]==3
    assert result["gain_candidate_points"]==pytest.approx(100/3)
    assert score_pair(np.array([2,1,0]),probs_b,probs_c,
                      np.array([False]*3),ytrain,np.array([0,0,2])) is None


def test_previous_baseline_mismatch_rejects():
    old_metrics={"n":2,"correct":1,"neutral_errors":1,
                 "opposite_errors":0,"predicted_up":1,"predicted_down":1}
    old={"horizons":{"H2":{
        "ml_tuning":{"feasible":True,"threshold":0.2},
        "best_iteration":10,
        "train_counts":{"DOWN":2,"NEUTRAL":1,"UP":2},
        "rules":{"lightgbm_30":{
            "all_check":old_metrics,
            "shared_ml_selected":old_metrics}},
        "by_asset_selected":{
            sym:{"lightgbm_30":old_metrics}
            for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
    }}}
    row={"baseline":old_metrics}
    all_check={**row,"best_iteration":10,
               "train_counts":{"DOWN":2,"NEUTRAL":1,"UP":2}}
    asset={sym:row for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
    verify_original_fold(old,horizon=2,threshold={"feasible":True,"threshold":0.2},
                         all_check=all_check,selected=row,by_asset=asset)
    with pytest.raises(ValueError,match="threshold drift"):
        verify_original_fold(old,horizon=2,threshold={"feasible":True,"threshold":0.3},
                             all_check=all_check,selected=row,by_asset=asset)
    row_bad={"baseline":{**old_metrics,"correct":0}}
    with pytest.raises(ValueError,match="benchmark count drift"):
        verify_original_fold(old,horizon=2,threshold={"feasible":True,"threshold":0.2},
                             all_check=all_check,selected=row_bad,by_asset=asset)


def test_smoke_decision_cannot_approve_candidate():
    m={"n":10,"precision":0.4,"correct":4}
    c={"n":10,"precision":0.6,"correct":6}
    paired={"n":10,"baseline":m,"candidate":c,
            "gain_candidate_points":20.0}
    p={
        BASELINE:{"log_loss":1.0,"multiclass_brier_sum":0.6,
                  "movement_and_oracle_direction":{
                      "movement_detection":{"roc_auc":0.55}}},
        CANDIDATE:{"log_loss":0.9,"multiclass_brier_sum":0.5,
                   "movement_and_oracle_direction":{
                       "movement_detection":{"roc_auc":0.65}}},
    }
    fake=[]
    for i in range(2):
        h={}
        for hk in ("H2","H6"):
            h[hk]={"shared_baseline_selected":paired,
                   "by_asset_selected":{
                       sym:paired for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")},
                   "raw_probability_metrics_all_check":p}
        fake.append({"horizons":h})
    res=summarize(fake)
    assert res["H2"]["mean_gain_points"]==pytest.approx(20)
    assert res["H2"]["folds_candidate_wins"]==2
    assert res["decision"]=="TECHNICAL_SMOKE_NOT_SCIENTIFIC_RESULT"
