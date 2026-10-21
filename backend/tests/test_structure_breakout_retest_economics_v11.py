"""Synthetic only: V11 fixed economics, no historical market access."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.structure_breakout_retest_economics_v11 import (
    FIRST, CUTOFF, HORIZONS, score_symbol, _stats, report,
)
from mlpipeline.structure_breakout_retest_historical_v11 import detect as historical
from mlpipeline.structure_breakout_retest_v9 import detect as prospective


def candles(n=700,starting="2026-10-21T00:00:00Z"):
    stamps=pd.date_range(starting,periods=n,freq="5min")
    x=np.arange(n,dtype=float)
    prices=100+0.025*x+np.sin(x/5.0)*0.15
    return pd.DataFrame({
        "symbol":"BTCUSDT","bar_open_at":stamps,"bar_close_at":stamps+pd.Timedelta(minutes=5),
        "open":prices,"high":prices+0.35,"low":prices-0.35,"close":prices,
        "is_final":True})


def mocked_events(bars,at=450,side="LONG",atr=1.0):
    decision=bars.bar_close_at.iloc[at]
    return pd.DataFrame([{
        "symbol":"BTCUSDT","side":side,
        "breakout_close_at":decision-pd.Timedelta(minutes=5),
        "decision_close_at":decision,"atr_at_breakout":atr,
        "h1_regime":"UP" if side=="LONG" else "DOWN",
    }])


@pytest.mark.parametrize("side",["LONG","SHORT"])
def test_fixed_common_population_three_horizons(side):
    b=candles()
    events,count=score_symbol(b,mocked_events(b,side=side))
    assert count["candidate_decisions"]==count["matured"]==1
    assert len(events)==1
    event=events.iloc[0]
    assert event.entry_at==b.bar_open_at.iloc[451]
    for h in HORIZONS:
        assert np.isclose(event[f"h{h}_net_5bps"],event[f"h{h}_gross_bps"]-5)
        assert event[f"h{h}_exit_reason"] in (
            "gap_stop","gap_target_capped","both_hit_stop_first","stop","target","timeout")


def test_24bar_horizon_rejects_all_if_future_short():
    b=candles(n=480)
    at=470
    events,count=score_symbol(b,mocked_events(b,at=at))
    assert events.empty
    assert {"h6_gross_bps","h12_gross_bps","h24_gross_bps"}.issubset(events.columns)
    assert count["no_contiguous_full_24bar_window"]==1
    assert count["matured"]==0
    assert _stats(events,12)["trades"]==0


def test_reject_gap_without_synthetic_ohlc():
    b=candles()
    b.loc[452:,"bar_open_at"]+=pd.Timedelta(minutes=5)
    b.loc[452:,"bar_close_at"]+=pd.Timedelta(minutes=5)
    events,count=score_symbol(b,mocked_events(b))
    assert events.empty
    assert count["no_contiguous_full_24bar_window"]==1


def test_reject_maturity_at_historical_cutoff():
    b=candles(n=400,starting="2026-06-29T16:00:00Z")
    at=390
    events,count=score_symbol(b,mocked_events(b,at=at))
    assert events.empty
    assert count["no_contiguous_full_24bar_window"]==1


def test_cost_corruption_rejected():
    b=candles()
    events,_=score_symbol(b,mocked_events(b))
    events.loc[0,"h12_net_5bps"]=10000.0
    with pytest.raises(AssertionError,match="Costs incorrect"):
        _stats(events,12)


def test_historical_clone_same_later_decisions_as_prospective():
    b=candles(n=800)
    a=historical(b,start=pd.Timestamp("2026-10-22T00:00:00Z"))
    c=prospective(b,start=pd.Timestamp("2026-10-22T00:00:00Z"))
    pd.testing.assert_frame_equal(a,c,check_like=True)


def test_no_forbidden_asset_or_fingerprint_ignored():
    b=candles()
    events,_=score_symbol(b,mocked_events(b))
    x={s:events.copy() for s in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
    x["ADAUSDT"]=x.pop("ETHUSDT")
    with pytest.raises(ValueError,match="four approved"):
        report(x,{s:{"matured":len(f),"candidate_decisions":len(f)}
                  for s,f in x.items()},"a"*64)
