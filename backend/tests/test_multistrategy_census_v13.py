"""Synthetic-only V13 tests. No returns, no protected months, no network."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.multistrategy_census_v13 import (
    STRATEGIES, aggregate, _features, _select, candidates,
)


def candles(n=2300):
    stamps=pd.date_range("2026-05-01T00:00:00Z",periods=n,freq="5min")
    x=np.arange(n,dtype=float)
    close=100+0.012*x+0.4*np.sin(x/19)
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":stamps,
        "bar_close_at":stamps+pd.Timedelta(minutes=5),
        "open":close-0.02,"high":close+0.25,"low":close-0.25,
        "close":close,"is_final":True})


def test_aggregate_uses_only_closed_aligned_blocks():
    f=candles(n=48)
    h1=aggregate(f,12)
    h4=aggregate(f,48)
    assert len(h1)==4
    assert len(h4)==1
    assert h1.bar_close_at.iloc[0]==pd.Timestamp("2026-05-01T01:00:00Z")
    f=f.drop(index=[4]).reset_index(drop=True)
    assert len(aggregate(f,12))==3
    assert aggregate(f,48).empty


def test_future_mutation_does_not_change_past_candidates():
    f=candles()
    original=candidates(f)
    newer=f.copy()
    # Change only bars after a cut without changing OHLC validity.
    boundary=2100
    newer.loc[boundary:,"open"]*=1.05
    newer.loc[boundary:,"close"]*=1.05
    newer.loc[boundary:,"high"]*=1.08
    newer.loc[boundary:,"low"]*=0.98
    after=candidates(newer)
    limit=f.bar_close_at.iloc[boundary-1]
    old=original.loc[original.decision_close_at.le(limit)].reset_index(drop=True)
    new=after.loc[after.decision_close_at.le(limit)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old,new)


def test_ohlc_finality_and_symbol():
    f=candles()
    f.loc[0,"is_final"]=False
    with pytest.raises(ValueError,match="closed"):
        candidates(f)


def test_all_strategies_are_declared_and_no_duplicate_or_future_signals():
    f=candles()
    signals=candidates(f)
    assert signals.strategy.isin(STRATEGIES).all()
    assert not signals.duplicated(["symbol","strategy","decision_close_at"]).any()
    if not signals.empty:
        assert signals.decision_close_at.isin(f.bar_close_at).all()
        assert signals.side.isin(("LONG","SHORT")).all()
        assert signals.reference_price.gt(0).all()


def test_h1_gap_resets_all_indicators_and_does_not_impute():
    f=candles(n=2400)
    # The dropped bar removes an entire H1, so features must restart after a gap.
    f=f.drop(index=[600]).reset_index(drop=True)
    h1=aggregate(f,12)
    result=_features(h1,60)
    after_gap=result.loc[result.bar_open_at.diff().gt(pd.Timedelta(hours=1))]
    assert len(after_gap)>=1
    assert after_gap.atr14.isna().all()
    assert after_gap.d_high20.isna().all()
