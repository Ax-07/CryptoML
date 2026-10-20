"""Synthetic-only V2 tests; no historical labels, reserved data or model fitting."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.pullback_regime_v2 import describe_regimes,regime_census,FEATURE_COLUMNS

def candles(n=140):
    end=pd.date_range("2026-02-01T00:05:00Z",periods=n,freq="5min")
    close=100+0.1*np.arange(n)
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":end-pd.Timedelta(minutes=5),
       "bar_close_at":end,"open":close-0.03,"high":close+0.1,
       "low":close-0.15,"close":close,"is_final":True})

def test_descriptive_features_are_causal():
    bars=candles()
    first=describe_regimes(bars).iloc[:81].reset_index(drop=True)
    changed=bars.copy()
    changed.loc[81:,["open","high","low","close"]]*=2
    after=describe_regimes(changed).iloc[:81].reset_index(drop=True)
    pd.testing.assert_frame_equal(first,after)
    assert set(FEATURE_COLUMNS).issubset(first.columns)

def test_gap_resets_and_does_not_leak_prior_values():
    bars=candles()
    bars.loc[81:,"bar_open_at"]+=pd.Timedelta(minutes=5)
    bars.loc[81:,"bar_close_at"]+=pd.Timedelta(minutes=5)
    f=describe_regimes(bars)
    assert f.segment_id.iloc[80]!=f.segment_id.iloc[81]
    assert pd.isna(f.v2_trend_spread_atr.iloc[81])
    assert pd.isna(f.v2_ema9_slope_5_atr.iloc[81])

def test_flat_zero_range_has_missing_close_position():
    bars=candles()
    bars.loc[80,["open","high","low","close"]]=bars.loc[80,"close"]
    f=describe_regimes(bars)
    assert pd.isna(f.v2_pullback_close_position.iloc[80])

def test_census_reconciles_and_keeps_zero():
    f=describe_regimes(candles())
    f.loc[:,"pullback_signal"]=False
    assert regime_census(f)["events"]==0
    f.loc[80,"pullback_signal"]=True
    report=regime_census(f)
    assert report["events"]==1
    assert sum(report["strength"].values())==report["valid_events"]
    assert sum(report["depth"].values())==report["valid_events"]

def test_missing_columns_fails():
    f=describe_regimes(candles())
    with pytest.raises(ValueError,match="Missing"):
        regime_census(f.drop(columns=[FEATURE_COLUMNS[0]]))
