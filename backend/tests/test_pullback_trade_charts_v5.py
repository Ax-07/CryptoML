"""Synthetic-only visualization tests; no market/P3/reserved data."""
import pandas as pd
import numpy as np
import pytest

from mlpipeline.pullback_trade_charts_v5 import actual_exit_time,_svg


def scenario(reason):
    t=pd.Timestamp("2026-03-01T00:05:00Z")
    opens=pd.date_range(t-pd.Timedelta(minutes=5),periods=26,freq="5min")
    bars=pd.DataFrame({
        "bar_open_at":opens,
        "bar_close_at":opens+pd.Timedelta(minutes=5),
        "open":100., "high":100.3, "low":99.7,"close":100.,
    })
    i=10
    signal=bars.bar_close_at.iloc[i]
    entry=signal+pd.Timedelta(minutes=5)
    entry_i=i+2
    if reason=="target":
        bars.loc[entry_i,"high"]=101.6
        price=101.5
    elif reason=="stop":
        bars.loc[entry_i,"low"]=98.8
        price=99.
    else:
        price=100.
    event=pd.Series({"signal_close_at":signal,"entry_at":entry,
                     "mature_at":signal+pd.Timedelta(minutes=35),
                     "entry_price":100.,"exit_price":price,
                     "atr_at_signal":1.,"exit_reason":reason,
                     "gross_bps":10000*(price/100.-1),"net_5bps":10000*(price/100.-1)-5,
                     "symbol":"BTCUSDT"})
    return bars,event


@pytest.mark.parametrize("reason",["target","stop","timeout"])
def test_replay_and_svg_are_faithful(reason):
    bars,event=scenario(reason)
    exit_at,stop,target=actual_exit_time(bars,event)
    assert stop==99.
    assert target==101.5
    if reason=="timeout":
        assert exit_at==event.mature_at
    else:
        assert exit_at==bars.bar_close_at.iloc[12]
    svg,marked_exit=_svg(bars,event)
    assert marked_exit==exit_at
    assert "<svg" in svg and "SIGNAL" in svg and "TARGET" in svg


def test_reject_mismatched_exit():
    bars,event=scenario("stop")
    event["exit_reason"]="target"
    with pytest.raises(AssertionError,match="Replayed exit differs"):
        actual_exit_time(bars,event)
