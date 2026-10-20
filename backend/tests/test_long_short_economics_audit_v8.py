"""Synthetic P4 V8 barrier symmetry, gaps, clock and partition arithmetic."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.long_short_economics_audit_v8 import (
    _barrier, _stats, _group_report, symbol_events,
    verify_legacy_long,
)
from mlpipeline.market_structure_v6 import classify_m5


def windows(side,hit,entry=100.,atr=1.):
    start=pd.date_range("2026-05-04T00:00:00Z",periods=6,freq="5min")
    f=pd.DataFrame({"open":np.full(6,entry),"high":np.full(6,entry+0.2),
                    "low":np.full(6,entry-0.2),"close":np.full(6,entry),
                    "bar_open_at":start,
                    "bar_close_at":start+pd.Timedelta(minutes=5)})
    s=1 if side=="LONG" else -1
    if hit=="stop":
        if s==1:f.loc[0,"low"]=entry-atr-0.1
        else:f.loc[0,"high"]=entry+atr+0.1
    if hit=="target":
        if s==1:f.loc[0,"high"]=entry+1.5*atr+0.1
        else:f.loc[0,"low"]=entry-1.5*atr-0.1
    if hit=="both_hit_stop_first":
        f.loc[0,"high"]=entry+2*atr;f.loc[0,"low"]=entry-2*atr
    if hit=="gap_stop":
        f.loc[0,"open"]=entry-s*1.1*atr
        f.loc[0,"high"]=max(f.loc[0,"open"],f.loc[0,"high"])
        f.loc[0,"low"]=min(f.loc[0,"open"],f.loc[0,"low"])
    if hit=="gap_target_capped":
        f.loc[0,"open"]=entry+s*1.6*atr
        f.loc[0,"high"]=max(f.loc[0,"open"],f.loc[0,"high"])
        f.loc[0,"low"]=min(f.loc[0,"open"],f.loc[0,"low"])
    return f


@pytest.mark.parametrize("side",["LONG","SHORT"])
@pytest.mark.parametrize("reason",[
    "stop","target","timeout","both_hit_stop_first","gap_stop","gap_target_capped"
])
def test_mirrored_barriers(side,reason):
    win=windows(side,reason)
    exit_price,actual,gross=_barrier(win,100.,1.,side)
    assert actual==reason
    sign=1 if side=="LONG" else -1
    expected={"stop":-1.,"target":1.5,"timeout":0.,
              "both_hit_stop_first":-1.,"gap_stop":-1.1,
              "gap_target_capped":1.5}[reason]
    assert np.isclose(exit_price,100+sign*expected)
    assert np.isclose(gross,100*expected)


def test_filter_partition_and_fees():
    values=[10.,-15.,5.]
    f=pd.DataFrame({
        "gross_bps":values,"net_2bps":np.array(values)-2,
        "net_5bps":np.array(values)-5,"net_10bps":np.array(values)-10,
        "confirmed_B":[True,False,True],"aligned":[True,False,False],
        "exit_reason":["target","stop","timeout"],
    })
    r=_group_report(f)
    assert r["A"]["ALL"]["trades"]==3
    assert r["A"]["ALIGNED"]["trades"]==1
    assert r["A"]["REJECTED"]["trades"]==2
    assert r["B"]["ALL"]["trades"]==2
    assert r["B"]["ALIGNED"]["trades"]==1
    assert np.isclose(r["A"]["ALL"]["sum_net_5bps"],-15)
    assert r["A"]["ALL"]["net_5bps_per_opportunity"]==-5
    assert _stats(f.iloc[:0],3)["sum_net_5bps"]==0


def test_reject_bad_cost():
    f=pd.DataFrame({"gross_bps":[2.],"exit_reason":["timeout"],
        "net_2bps":[0.],"net_5bps":[100.],"net_10bps":[-8.]})
    with pytest.raises(AssertionError,match="round-trip"):
        _stats(f,1)


def test_event_windows_do_not_access_incomplete_future():
    n=270
    t=pd.date_range("2026-06-30T01:00:00Z",periods=n,freq="5min")
    p=100+np.arange(n)*0.02+np.sin(np.arange(n)/2)*0.1
    bars=pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":t,
        "bar_close_at":t+pd.Timedelta(minutes=5),
        "open":p,"high":p+0.3,"low":p-0.3,"close":p,
        "is_final":True})
    classification=classify_m5(bars)
    events,counts=symbol_events(bars,classification)
    assert counts["candidate_signals"]==counts["matured"]+counts["excluded_immature_or_gap_or_invalid"]
    if not events.empty:
        assert events.mature_at.lt(pd.Timestamp("2026-07-01T00:00:00Z")).all()
        assert events.entry_at.eq(events.signal_close_at+pd.Timedelta(minutes=5)).all()
        assert events.mature_at.eq(events.signal_close_at+pd.Timedelta(minutes=35)).all()
