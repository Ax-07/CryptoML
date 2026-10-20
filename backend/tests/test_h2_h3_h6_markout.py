"""Synthetic tests, no P0, no network."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.horizon_markout_source import phase_rows,evaluate_rows,HORIZONS,MODELS

def sample():
    n=12
    x=pd.DataFrame({
        "fold_id":[0]*n,"symbol":["BTCUSDT"]*n,"horizon":[3]*n,
        "bar_close_at":pd.date_range("2025-01-01",periods=n,freq="5min",tz="UTC"),
        "actual_return":[.001,-.002,0.]*4
    })
    for name in MODELS:
        x[name+"_sign"]=np.where(np.arange(n)%3==1,0,2)
    x["always_down_sign"]=0
    x["reversion_1_sign"]=2
    return x

def test_fixed_phases_do_not_overlap():
    x=sample()
    segments=[phase_rows(x,p) for p in range(3)]
    assert [len(s) for s in segments]==[4,4,4]
    assert sum(len(s) for s in segments)==len(x)
    assert all(s.bar_close_at.diff().dropna().eq(pd.Timedelta(minutes=15)).all() for s in segments)

def test_cost_and_flats():
    x=evaluate_rows(sample())
    assert x["n"]==12 and x["nonflat"]==8 and x["flat"]==4
    m=x["models"]["lightgbm_30"]
    assert m["accuracy_nonflat"]==1.
    assert m["gross_mean_bps"]==pytest.approx(10.)
    assert m["net_mean_bps"]["2"]==pytest.approx(8.)
    assert m["net_mean_bps"]["10"]==pytest.approx(0.)

def test_phase_rejects_mixed_horizons():
    s=sample()
    s.loc[0,"horizon"]=6
    with pytest.raises(ValueError):
        phase_rows(s,0)

def test_supported_horizons():
    assert HORIZONS==(2,3,6)
