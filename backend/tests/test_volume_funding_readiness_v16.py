"""V16 tests: no network, P0, or economic outcomes."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.volume_funding_readiness_v16 import census_symbol,_validate_volume,_aggregate_m15


def candles(n=250,side=1):
    t=pd.date_range("2025-04-01T00:00:00Z",periods=n,freq="5min")
    x=np.arange(n,dtype=float)
    p=100+side*x*0.05
    volume=np.full(n,10.)
    buy=np.full(n,5.)
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":t,
      "bar_close_at":t+pd.Timedelta(minutes=5),"open":p,
      "high":p+0.3,"low":p-0.3,"close":p,"is_final":True,
      "volume_base":volume,"volume_quote":volume*p,
      "taker_buy_volume_base":buy,"taker_buy_volume_quote":buy*p,
      "trade_count":np.full(n,30)})


def test_m15_complete_and_gap_never_imputed():
    c=candles(48)
    assert len(_aggregate_m15(c))==16
    c=c.drop(index=[4]).reset_index(drop=True)
    assert len(_aggregate_m15(c))==15


def test_invalid_taker_volume_rejected():
    c=candles()
    c.loc[0,"taker_buy_volume_base"]=20.
    with pytest.raises(ValueError,match="exceeds total"):
        _validate_volume(c)


def test_partial_candle_rejected():
    c=candles()
    c.loc[0,"is_final"]=False
    with pytest.raises(ValueError,match="closed"):
        census_symbol(c)


def test_later_changes_leave_prior_signals_untouched():
    c=candles(450)
    c.loc[250:252,["volume_base","taker_buy_volume_base"]]=[100.,100.]
    a,_=census_symbol(c)
    d=c.copy()
    d.loc[350:,"close"]+=2.
    d.loc[350:,"high"]+=2.
    d.loc[350:,"open"]+=2.
    d.loc[350:,"low"]+=2.
    d.loc[350:,"volume_base"]*=2
    d.loc[350:,"taker_buy_volume_base"]*=2
    d.loc[350:,"volume_quote"]*=2
    d.loc[350:,"taker_buy_volume_quote"]*=2
    b,_=census_symbol(d)
    cutoff=c.bar_close_at.iloc[349]
    aa=a.loc[a.decision_close_at.le(cutoff)].reset_index(drop=True)
    bb=b.loc[b.decision_close_at.le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(aa,bb)


def test_symmetric_high_volume_signal():
    for side in (1,-1):
        c=candles(360,side=side)
        # All the last three M5 of a completed M15 are impulsive.
        c.loc[240:242,"volume_base"]=100.
        c.loc[240:242,"taker_buy_volume_base"]=100. if side==1 else 0.
        c.loc[240:242,"volume_quote"]=c.loc[240:242,"volume_base"]*c.loc[240:242,"close"]
        c.loc[240:242,"taker_buy_volume_quote"]=c.loc[240:242,"taker_buy_volume_base"]*c.loc[240:242,"close"]
        signals,counts=census_symbol(c)
        assert len(signals)>=1
        assert (signals.side=="LONG" if side==1 else signals.side=="SHORT").any()
