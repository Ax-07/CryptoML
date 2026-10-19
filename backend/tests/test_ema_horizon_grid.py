"""Synthetic security and functional tests: H1..H6 EMA experiment; no P0."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.ema_horizon_grid import (
    HORIZONS, EMA_COLUMNS, ema_features, fold_examples,
    rule_mask, rule_sign, score_next_bar, sign,
)
from mlpipeline.features import FEATURE_COLUMNS, build_symbol_features
from mlpipeline.labels import add_labels


def bars(n=80, gap_at=None, flat=False, scale=1):
    times=pd.date_range("2025-01-01",periods=n,freq="5min",tz="UTC")
    if gap_at is not None:
        times=pd.DatetimeIndex([t+(pd.Timedelta(minutes=5) if i>=gap_at
                                    else pd.Timedelta(0))
                                for i,t in enumerate(times)])
    value=(100+np.arange(n,dtype=float)*0.1)*scale
    if flat:
        value[:]=100*scale
    return pd.DataFrame({
        "symbol":"BTCUSDT","bar_open_at":times,
        "bar_close_at":times+pd.Timedelta(minutes=5),
        "open":value,"high":value+0.05*scale,
        "low":value-0.05*scale,"close":value,
        # Complete synthetic OHLCV contract required by build_symbol_features.
        # No real data; volume is deterministic and positive.
        "volume_base":np.full(n,10.0),
        "volume_quote":10.0*value,
        "trade_count":np.full(n,100,dtype=int),
        "taker_buy_volume_base":np.full(n,5.0),
    })


def test_all_horizons_and_price_invariance():
    assert HORIZONS==(1,2,3,4,5,6)
    assert len(FEATURE_COLUMNS)==30
    assert EMA_COLUMNS==("ema9_price_gap","ema9_ema26_gap")
    a=ema_features(bars())
    b=ema_features(bars(scale=200))
    np.testing.assert_allclose(
        a[list(EMA_COLUMNS)].to_numpy(dtype=float),
        b[list(EMA_COLUMNS)].to_numpy(dtype=float),atol=1e-6,equal_nan=True)
    assert a[list(EMA_COLUMNS)].iloc[:25].isna().any(axis=None)
    assert np.isfinite(a.loc[25,list(EMA_COLUMNS)].to_numpy(dtype=float)).all()


def test_prefix_invariance_and_gap_reset():
    raw=bars(80)
    full=ema_features(raw)
    for n in (1,9,25,26,42,79):
        prefix=ema_features(raw.iloc[:n])
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True),
                                      full.iloc[:n].reset_index(drop=True))
    changed=ema_features(bars(80,gap_at=42))
    assert changed.loc[42:66,list(EMA_COLUMNS)].isna().any(axis=None)
    assert np.isfinite(changed.loc[67,list(EMA_COLUMNS)].to_numpy(dtype=float)).all()


def test_H1_is_exact_next_bar_and_other_horizons_are_cumulative():
    f=build_symbol_features(bars(95))
    for h in HORIZONS:
        labels=add_labels(f,horizon=h,threshold_multiplier=0.5)
        i=60
        assert labels.loc[i,"label_mature_at"]==f.loc[i,"bar_close_at"]+pd.Timedelta(minutes=5*h)
        expected=f.loc[i+h,"close"]/f.loc[i,"close"]-1
        assert labels.loc[i,"future_return"]==pytest.approx(expected)
    pred=np.array([2,0,2,0])
    measure=score_next_bar(
        np.array([0.01,-0.1,0.,0.5]),pred,np.ones(4,dtype=bool))
    assert measure["flat"]==1
    assert measure["nonflat"]==3
    assert measure["correct_nonflat"]==2
    assert measure["sign_accuracy"]==pytest.approx(2/3)


def test_rule_selection_uses_unlabeled_tuning_and_no_future_truth():
    tune=np.linspace(-0.02,0.02,30)
    check=np.array([0.,0.1,-0.1])
    mask,config=rule_mask(tune,check)
    assert mask.tolist()==[False,True,True]
    assert config["target_tuning_fraction"]==0.4
    assert rule_sign(np.array([-0.1,0.,0.2]),majority=0).tolist()==[0,0,2]
    assert sign(np.array([[0.1,0.8,0.1],[0.2,0.4,0.4]]),0).tolist()==[0,2]


def test_ema_rejects_corrupt_time():
    x=bars()
    x.loc[10,"bar_close_at"]+=pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match="Invalid closed"):
        ema_features(x)


def test_next_bar_flat_excludes_denominator():
    r=score_next_bar(np.array([0.,0.,0.]),np.array([0,2,0]),
                     np.ones(3,dtype=bool))
    assert r["nonflat"]==0
    assert r["flat"]==3
    assert r["sign_accuracy"] is None
