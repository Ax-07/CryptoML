"""Synthetic-only H2 selective forensic audit tests, never P0 or live orders."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.h2_selective_concentration_audit import audit,verified,aggregate,describe,extremes
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS

def sample():
    frame=pd.DataFrame({
        "fold_id":[1,1,2,2],
        "symbol":["BTCUSDT","BTCUSDT","ETHUSDT","ETHUSDT"],
        "bar_close_at":pd.date_range("2025-01-01",periods=4,freq="5min",tz="UTC"),
        "delayed_return_bps":[10.,-5.,7.,-2.],
        "regression_30_expected_bps":[4.,-4.,4.,0.],
        "regression_30_side":[1,-1,1,0],
        "ridge_ema2_expected_bps":[0.]*4,
        "ridge_ema2_side":[0]*4,
        "reference_3class_H2_30_side":[1]*4,
    })
    frame["delayed_mature_at"]=frame.bar_close_at+pd.Timedelta(minutes=10)
    return frame

def test_fold_out_and_cost_accounting():
    report=audit(sample())
    assert report["all"]["trades"]==3
    assert report["all"]["sum_net_bps"]==pytest.approx(16.)
    assert report["fold_1"]["sum_net_bps"]==pytest.approx(11.)
    assert report["without_fold_1"]["sum_net_bps"]==pytest.approx(5.)
    assert report["positive_folds"]==2
    assert report["empty_folds"]==34

def test_fixed_extreme_sensitivity():
    trade=describe(sample())
    x=extremes(trade)
    assert x["best_five"][0]["net_bps"]==pytest.approx(8.)
    assert x["quantiles_net_bps"]["0"]==pytest.approx(3.)
    assert x["outlier_sensitivity"][0]["without_best"]["sum_net_bps"]==pytest.approx(8.)

def test_reject_bad_protocol():
    with pytest.raises(ValueError):
        verified(sample(),{},[])

def test_consecutive_signal_run_length():
    out=audit(sample())
    assert out["longest_consecutive_signal_runs"][0]["signals_consecutive"]==2
