"""Synthetic-only H1 binary benchmark tests (no P0, REST or reserved periods)."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.h1_binary_diagnostic import (
    PROTOCOL,CANDIDATES, nonflat_sign, check_metrics, _summarize, fit_candidates,
)


def check_rows():
    return pd.DataFrame({
        "symbol":["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"],
        "bar_close_at":pd.date_range("2025-01-01",periods=4,freq="5min",tz="UTC"),
        "future_return_h1":[.01,-.02,0.,.03],
        "ret_1":[-.002,.003,-.004,0.],
        "baseline_shared_mask":[True,True,True,False],
        "ema2_autonomous_mask":[True,False,True,True],
        "lightgbm_ema2_sign":[2,0,0,0],
        "lightgbm_30_sign":[0,0,2,2],
    })


def test_nonflat_is_explicit_no_down_zero():
    y,keep=nonflat_sign(np.array([0.,.002,-.001,0.,.03]))
    assert y.tolist()==[1,0,1]
    assert keep.tolist()==[False,True,True,False,True]
    with pytest.raises(ValueError):
        nonflat_sign(np.array([np.nan]))


def test_candidate_specification_fixed():
    assert list(CANDIDATES)==["binary_ema2","binary_ema2_close",
                              "binary_30","logistic_ema2"]
    assert len(CANDIDATES["binary_ema2"])==2
    assert len(CANDIDATES["binary_ema2_close"])==3
    assert len(CANDIDATES["binary_30"])==30


def test_paired_scores_use_identical_nonflat_rows_and_flat_excluded():
    check=check_rows()
    probs={
        "binary_ema2":[.7,.4,.8,.2],
        "binary_ema2_close":[.6,.3,.2,.8],
        "binary_30":[.2,.6,.8,.8],
        "logistic_ema2":[.5,.5,.8,.5],
    }
    metrics,pred=check_metrics(check,probs)
    assert metrics["n_all"]==4 and metrics["n_nonflat"]==3
    assert metrics["flat"]==1
    assert metrics["accuracy"]["binary_ema2"]["correct"]==2
    assert metrics["accuracy"]["binary_ema2_close"]["correct"]==3
    assert metrics["accuracy"]["lightgbm_ema2"]["correct"]==2
    assert metrics["accuracy"]["always_down"]["correct"]==1
    assert metrics["accuracy"]["logistic_ema2"]["correct"]==1
    assert metrics["views"]["original_30_selected"]["n"]==2
    pair=metrics["paired"]["binary_ema2_close_minus_lightgbm_ema2"]
    assert pair["candidate_only_correct"]==1
    assert pair["reference_only_correct"]==0
    assert len(pred)==4 and pred["is_flat"].sum()==1
    assert np.isnan(pred.loc[2,"actual_up_or_null"])


def test_training_fails_closed_on_matured_nonflat_requirement():
    x=pd.DataFrame({
        "future_return_h1":np.zeros(102),
        "ema9_price_gap":np.zeros(102),"ema9_ema26_gap":np.zeros(102),
        "close_location":np.zeros(102),
    })
    with pytest.raises(ValueError,match="Insufficient"):
        fit_candidates(x,x,x)


def test_summary_fold_counts():
    c=check_rows()
    c["fold_id"]=0
    m,_=check_metrics(c,{
        "binary_ema2":[.7,.4,.8,.2],
        "binary_ema2_close":[.6,.3,.2,.8],
        "binary_30":[.2,.6,.8,.8],
        "logistic_ema2":[.5,.5,.8,.5],
    })
    summary=_summarize([{"metrics":m}])
    assert summary["accuracy"]["binary_ema2_close"]["n"]==3
    assert summary["paired"]["binary_ema2_close_minus_lightgbm_ema2"]["positive_folds"]==1
    assert summary["flat_checks"]==1
