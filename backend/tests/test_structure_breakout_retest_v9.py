"""Synthetic-only V9 tests. No historical market outcomes or trading."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.structure_breakout_retest_v9 import (
    detect, PROSPECTIVE_START, _asof_regime,
)


def bars(n=560,symbol="BTCUSDT"):
    t=pd.date_range("2026-10-20T23:00:00Z",periods=n,freq="5min")
    x=np.arange(n)
    close=100+0.08*x+np.sin(x/8)*0.5
    return pd.DataFrame({"symbol":symbol,"bar_open_at":t,"bar_close_at":t+pd.Timedelta(minutes=5),
                         "open":close-0.02,"high":close+0.3,
                         "low":close-0.3,"close":close,"is_final":True})


def test_guard_rejects_historical_start_and_unauthorized_symbol():
    f=bars()
    with pytest.raises(ValueError,match="historical"):
        detect(f,start=pd.Timestamp("2026-06-01T00:00:00Z"))
    with pytest.raises(ValueError,match="approved"):
        detect(bars(symbol="ADAUSDT"))


def test_guard_requires_completed_and_sufficient_candles():
    with pytest.raises(ValueError,match="400"):
        detect(bars(n=200))
    f=bars()
    f.loc[len(f)-1,"is_final"]=False
    with pytest.raises(ValueError,match="closed"):
        detect(f)


def test_h1_asof_uses_last_closed_only_and_stales():
    times=pd.Series(pd.to_datetime(["2026-10-22T01:55:00Z",
                                    "2026-10-22T02:00:00Z",
                                    "2026-10-22T02:30:00Z",
                                    "2026-10-22T03:00:00Z"],utc=True))
    context=pd.DataFrame({"bar_close_at":pd.to_datetime(["2026-10-22T02:00:00Z"],utc=True),
                          "regime":["UP"]})
    got=_asof_regime(times,context,pd.Timedelta(hours=1))
    assert list(got)==["UNKNOWN","UP","UP","UNKNOWN"]


def test_future_candle_perturbation_cannot_change_prior_decisions():
    f=bars(n=700)
    original=detect(f)
    modified=f.copy()
    cutoff=620
    modified.loc[cutoff:,"close"]*=1.04
    modified.loc[cutoff:,"high"]*=1.08
    modified.loc[cutoff:,"open"]*=1.03
    modified.loc[cutoff:,"low"]*=0.98
    again=detect(modified)
    boundary=f.bar_close_at.iloc[cutoff-1]
    # A future change cannot change any decision already emitted before it.
    a=original.loc[original.decision_close_at.le(boundary)].reset_index(drop=True)
    b=again.loc[again.decision_close_at.le(boundary)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)


def test_no_observation_precedes_candidate_start():
    f=bars(n=520)
    out=detect(f)
    assert out.empty or out.decision_close_at.ge(PROSPECTIVE_START).all()
    if not out.empty:
        assert out.breakout_close_at.lt(out.decision_close_at).all()
        assert out.status.eq("OFFLINE_REPLAY_NOT_LIVE_ATTESTED").all()
