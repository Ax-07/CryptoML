"""Synthetic causal regression guards for one/two/three confirmation bars."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_features import (
    STRUCTURE_COLUMNS, build_market_structure_features,
)
from mlpipeline.market_structure_confirmation_sensitivity import (
    RIGHTS, VARIANTS, signal_rates, summarize, build_confirmation_shared,
)


def bars(highs, *, gap_at=None):
    highs=np.asarray(highs,dtype=float)
    n=len(highs)
    times=pd.date_range("2026-01-01T00:00:00Z",periods=n,freq="5min")
    if gap_at is not None:
        times=pd.DatetimeIndex([
            t+(pd.Timedelta(minutes=5) if i>=gap_at else pd.Timedelta(0))
            for i,t in enumerate(times)
        ])
    return pd.DataFrame({
        "symbol":"BTCUSDT","bar_open_at":times,
        "bar_close_at":times+pd.Timedelta(minutes=5),
        "high":highs,"low":highs-2,"close":highs-1,
    })


def test_schema_preserved_and_distinct_variants():
    assert RIGHTS==(1,2,3)
    assert len(FEATURE_COLUMNS)==30
    assert len(STRUCTURE_COLUMNS)==12
    assert [len(x) for x in VARIANTS.values()]==[30,42,42,42]
    assert set(VARIANTS["baseline_30"])==set(FEATURE_COLUMNS)


def test_pivot_revealed_at_p_plus_r_and_never_earlier():
    # Index 4 is a strict pivot high vs two prior and three later bars.
    h=[3,4,5,6,12,9,8,7,6,8,9,7,6]
    for right in RIGHTS:
        x=build_market_structure_features(bars(h),confirmation_bars=right)
        age=x["structure_age_swing_high"]
        assert age.iloc[4+right]==pytest.approx(0)
        assert age.iloc[4+right-1]!=age.iloc[4+right-1]  # NaN before confirmation
        assert age.iloc[5+right]==pytest.approx(1/48)


def test_r2_legacy_default_exactly_unchanged():
    rng=np.random.default_rng(514)
    h=100+rng.normal(0,5,100)
    baseline=build_market_structure_features(bars(h))
    explicit=build_market_structure_features(bars(h),confirmation_bars=2)
    pd.testing.assert_frame_equal(baseline,explicit)


@pytest.mark.parametrize("right",[1,2,3])
def test_prefix_invariance_each_confirmation(right):
    rng=np.random.default_rng(303+right)
    complete=bars(100+rng.normal(0,4,120))
    whole=build_market_structure_features(complete,confirmation_bars=right)
    for n in (3,8,13,37,94,119):
        prefix=build_market_structure_features(
            complete.iloc[:n].copy(),confirmation_bars=right)
        pd.testing.assert_frame_equal(
            prefix,whole.iloc[:n].reset_index(drop=True)
        )


@pytest.mark.parametrize("right",[1,2,3])
def test_gaps_reset_swing_state(right):
    h=[5,6,13,9,8,7,6,8,9,11,6,7,12,8,7,9,6]
    x=build_market_structure_features(bars(h,gap_at=10),confirmation_bars=right)
    assert pd.isna(x.loc[10,"structure_distance_swing_high"])
    assert x.loc[10,"structure_hh"]==0.
    assert x.loc[10,"structure_bos_up"]==0.


@pytest.mark.parametrize("bad",[0,4,-1,2.0,True])
def test_rejects_outside_declared_confirmation_grid(bad):
    with pytest.raises(ValueError,match="confirmation_bars"):
        build_market_structure_features(bars([1,2,3,4,5]),confirmation_bars=bad)


def test_event_frequency_on_common_rows():
    frame=pd.DataFrame()
    for r in RIGHTS:
        for col in ("structure_bos_up","structure_bos_down",
                    "structure_choch_up","structure_choch_down"):
            frame[f"r{r}_{col}"]=np.array([0,0,0,0],dtype="float32")
        frame.loc[0,f"r{r}_structure_bos_up"]=1.
        frame.loc[1,f"r{r}_structure_choch_down"]=1.
    freq=signal_rates(frame)
    for r in RIGHTS:
        assert freq[f"r{r}"]["bos_count"]==1
        assert freq[f"r{r}"]["choch_count"]==1
        assert freq[f"r{r}"]["any_break_count"]==2


def test_ada_rejected_before_any_feature_calculation():
    with pytest.raises(ValueError,match="ADA"):
        build_confirmation_shared({"ADAUSDT":bars([1,2,3,2,1,2])})


def test_paired_summary_does_not_count_infeasible():
    rows=[]
    outcomes={}
    for horizon in ("H2","H6"):
        outcomes[horizon]={}
        for key in VARIANTS:
            valid=key!="structure_r3_42"
            outcomes[horizon][key]={
                "feasible":valid,
                "check_metrics":{
                    "uplift_points":4. if key=="baseline_30" else 4.2,
                    "coverage":0.4,"raw_log_loss":1.06,
                } if valid else None,
            }
    rows.append({"horizons":outcomes})
    summary=summarize(rows)
    assert summary["H2"]["paired_vs_baseline"]["structure_r1_42"][
        "mean_uplift_delta_points"]==pytest.approx(0.2)
    assert summary["H6"]["paired_vs_baseline"]["structure_r3_42"][
        "paired_feasible_folds"]==0
