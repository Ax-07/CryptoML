"""Synthetic-only A/B comparison tests."""
import pandas as pd
import pytest
from mlpipeline.pullback_baselines import annotate_confirmation, summarize_prepared, _score

def series():
    ts=pd.date_range("2026-01-01T00:05:00Z",periods=10,freq="5min")
    bars=pd.DataFrame({"bar_open_at":ts-pd.Timedelta(minutes=5),"bar_close_at":ts,"open":[100]*10,"close":[100]*10})
    detected=pd.DataFrame({"bar_close_at":ts,"ema9":[99]*10,
                           "pullback_signal":[i in (2,6) for i in range(10)]})
    bars.loc[3,"close"]=101
    bars.loc[6,"close"]=102
    bars.loc[7,"open"]=103
    bars.loc[7,"close"]=102
    outcome=pd.DataFrame({"signal_close_at":[ts[2],ts[6]],
        "entry_at":[ts[2]+pd.Timedelta(minutes=5),ts[6]+pd.Timedelta(minutes=5)],
        "mature_at":[ts[2]+pd.Timedelta(minutes=35),ts[6]+pd.Timedelta(minutes=35)],
        "gross_bps":[12.,-6.],"net_2bps":[10.,-8.],"net_5bps":[7.,-11.],
        "net_10bps":[2.,-16.],"exit_reason":["target","stop"]})
    return bars,detected,outcome

def test_confirmation_only_uses_next_closed_bar():
    bars,features,outcomes=series()
    assert list(annotate_confirmation(bars,features,outcomes).baseline_b_confirmed)==[True,False]
    bars.loc[4,"close"]=900
    bars.loc[9,"close"]=500
    assert list(annotate_confirmation(bars,features,outcomes).baseline_b_confirmed)==[True,False]

def test_reject_wrong_entry_time():
    bars,features,outcomes=series()
    outcomes.loc[0,"entry_at"]+=pd.Timedelta(minutes=5)
    with pytest.raises(ValueError,match="Inconsistent confirmation"):
        annotate_confirmation(bars,features,outcomes)

def test_common_opportunities_and_monthly_reconciliation():
    bars,features,outcomes=series()
    events=annotate_confirmation(bars,features,outcomes)
    dataset={symbol:events.copy() for symbol in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
    report,monthly=summarize_prepared(dataset,"a"*64)
    a=report["overall"]["A_take_every_matured_pullback"]
    b=report["overall"]["B_confirm_next_close_before_common_entry"]
    assert a["trades"]==8 and b["trades"]==4
    assert a["opportunities"]==b["opportunities"]==8
    assert b["coverage"]==0.5
    assert a["net_2bps_per_opportunity"]==1.
    assert b["net_2bps_per_opportunity"]==5.
    assert len(monthly)==8

def test_zero_selected_is_valid():
    bars,features,outcomes=series()
    empty=annotate_confirmation(bars,features,outcomes).iloc[:0]
    result=_score(empty,total_opportunities=10)
    assert result["net_2bps_per_opportunity"]==0.
    assert result["net_2bps_per_trade"] is None

def test_reject_future_or_duplicated_outcomes():
    bars,features,outcomes=series()
    future=outcomes.copy()
    future.loc[0,"mature_at"]=pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError,match="out of P0 scope"):
        annotate_confirmation(bars,features,future)
    dup=pd.concat([outcomes,outcomes.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicate signal"):
        annotate_confirmation(bars,features,dup)
