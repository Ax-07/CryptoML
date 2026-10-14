"""Causal P2 structure regression guards; synthetic bars, no network/P0."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_features import (
    LEFT, RIGHT, STRUCTURE_COLUMNS, build_market_structure_features,
)
from mlpipeline.market_structure_diagnostics import VARIANTS, _summary


def bars(highs, lows=None, *, gap_at=None):
    n=len(highs)
    highs=np.asarray(highs,dtype=float)
    lows=np.asarray(lows if lows is not None else highs-2,dtype=float)
    closes=(highs+lows)/2
    times=pd.date_range("2026-01-01T00:00:00Z",periods=n,freq="5min")
    if gap_at is not None:
        times=pd.DatetimeIndex([
            stamp+(pd.Timedelta(minutes=5) if i>=gap_at else pd.Timedelta(0))
            for i,stamp in enumerate(times)
        ])
    return pd.DataFrame({
        "symbol":"BTCUSDT","bar_open_at":times,
        "bar_close_at":times+pd.Timedelta(minutes=5),
        "high":highs,"low":lows,"close":closes,
    })


def test_twelve_features_schema_does_not_touch_frozen_30():
    assert LEFT==RIGHT==2
    assert len(STRUCTURE_COLUMNS)==12
    assert len(FEATURE_COLUMNS)==30
    assert len(VARIANTS["baseline_30"])==30
    assert len(VARIANTS["baseline_plus_structure_42"])==42
    assert len(VARIANTS["structure_only_12"])==12
    assert not set(STRUCTURE_COLUMNS)&set(FEATURE_COLUMNS)


def test_pivot_confirmed_only_two_closed_bars_after_candidate():
    highs=[8,9,12,10,8,9,13,9,8,9,7,8]
    x=build_market_structure_features(bars(highs))
    assert pd.isna(x.loc[3,"structure_distance_swing_high"])
    assert x.loc[4,"structure_age_swing_high"]==pytest.approx(2/48)
    assert x.loc[7,"structure_hh"]==0
    assert x.loc[8,"structure_hh"]==1
    # A new higher swing is not retrospectively visible at its pivot index 6.
    assert x.loc[6,"structure_hh"]==0


def test_prefix_invariance_prevents_future_leakage():
    rng=np.random.default_rng(123)
    h=100+rng.normal(0,4,120)
    whole=bars(h)
    out=build_market_structure_features(whole)
    for n in (4,8,17,31,83,119):
        prefix=build_market_structure_features(whole.iloc[:n].copy())
        pd.testing.assert_frame_equal(
            prefix.reset_index(drop=True),
            out.iloc[:n].reset_index(drop=True),
        )


def test_future_spike_cannot_change_past_features():
    highs=[10,11,15,11,9,10,14,11,9,11,10,11]
    first=bars(highs)
    changed=first.copy()
    changed.loc[11,"high"]=100000
    changed.loc[11,"close"]=100
    a=build_market_structure_features(first)
    b=build_market_structure_features(changed)
    pd.testing.assert_frame_equal(a.iloc[:11],b.iloc[:11])


def test_m5_gap_resets_pivots_and_state():
    highs=[8,10,15,10,8,9,11,9,7,8,11,8,7]
    continuous=build_market_structure_features(bars(highs))
    with_gap=build_market_structure_features(bars(highs,gap_at=8))
    assert continuous.loc[8,"structure_hh"] in (0.,1.)
    assert with_gap.loc[8,"structure_hh"]==0.
    assert np.isnan(with_gap.loc[8,"structure_distance_swing_high"])
    assert with_gap.loc[8,"structure_bos_up"]==0


def test_zero_or_nonfinite_prices_fail_closed():
    invalid=bars([9,10,11,12,13])
    invalid.loc[3,"close"]=0
    with pytest.raises(ValueError,match="Invalid OHLC"):
        build_market_structure_features(invalid)


def test_ada_cannot_run_structure_comparison():
    from mlpipeline.market_structure_diagnostics import shared_with_structure
    x=bars([10,11,12,10,9,11,10])
    with pytest.raises(ValueError,match="ADA excluded"):
        shared_with_structure({"ADAUSDT":x})


def test_coverage_report_marks_infeasible_without_fabricating_score():
    fake=[]
    for i in range(2):
        outcomes={}
        for h in ("H2","H6"):
            outcomes[h]={
                variant:{"feasible":False,"check_metrics":None}
                for variant in VARIANTS
            }
        fake.append({"fold_id":i,"horizons":outcomes})
    result=_summary(fake)
    assert result["H2"]["baseline_30"]["feasible_folds"]==0
    assert result["H6"]["paired_vs_baseline"]["baseline_plus_structure_42"][
        "mean_uplift_delta_points"] is None
