"""Synthetic tests for V17 frozen A06 economics. No P0 or network."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.volume_taker_economics_v17 import score_symbol,summarize,validate_v16
from mlpipeline.volume_funding_readiness_v16 import PROTOCOL as V16_PROTOCOL,FIELDS
from mlpipeline.pullback_long_history_v3 import SYMBOLS


def bars(n=100, start="2026-06-01T00:00:00Z"):
    times=pd.date_range(start,periods=n,freq="5min")
    price=np.full(n,100.)
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":times,
        "bar_close_at":times+pd.Timedelta(minutes=5),
        "open":price,"high":price+0.1,"low":price-0.1,
        "close":price,"is_final":True})


def signal(b,side="LONG",at=30,atr=1.):
    return pd.DataFrame([dict(symbol="BTCUSDT",side=side,
        decision_close_at=b.bar_close_at.iloc[at],
        reference_close=100.,atr14_m15=atr,
        taker_imbalance=0.5 if side=="LONG" else -0.5,
        relative_volume=2.)])


@pytest.mark.parametrize("side",["LONG","SHORT"])
def test_score_symmetry_and_fees(side):
    b=bars()
    x,t=score_symbol(b,signal(b,side))
    assert len(x)==t[side]["matured"]==1
    assert x.entry_at.iloc[0]==b.bar_open_at.iloc[31]
    assert x.exit_reason.iloc[0]=="timeout"
    assert x.gross_bps.iloc[0]==pytest.approx(0.)
    for fee in (2,5,10):
        assert x[f"net_{fee}bps"].iloc[0]==pytest.approx(-fee)


def test_reject_future_immature_and_cutoff():
    b=bars()
    x,t=score_symbol(b,signal(b,at=94))
    assert x.empty and t["LONG"]["missing_or_immature"]==1
    b=bars(100,"2026-06-30T18:00:00Z")
    x,t=score_symbol(b,signal(b,at=63))
    assert x.empty and t["LONG"]["missing_or_immature"]==1


def test_gap_rejected():
    b=bars()
    b.loc[32:,"bar_open_at"]+=pd.Timedelta(minutes=5)
    b.loc[32:,"bar_close_at"]+=pd.Timedelta(minutes=5)
    x,t=score_symbol(b,signal(b))
    assert x.empty and t["LONG"]["missing_or_immature"]==1


def test_reject_invalid_atr():
    b=bars()
    x,t=score_symbol(b,signal(b,atr=0.))
    assert x.empty and t["LONG"]["invalid_atr_or_price"]==1


def test_v16_event_reference_parity():
    b=bars()
    item=signal(b)
    empty=item.iloc[:0].copy()
    series={s:item.copy() if s=="BTCUSDT" else empty.copy() for s in SYMBOLS}
    ref={"protocol":V16_PROTOCOL,"fingerprint_sha256":"a"*64,
         "source_partitions":84,"symbols":list(SYMBOLS),
         "total_signals":1,"by_symbol":{
             s:{"long":1 if s=="BTCUSDT" else 0,"short":0} for s in SYMBOLS}}
    validate_v16(ref,item.copy(),series,"a"*64)
    with pytest.raises(ValueError,match="identity"):
        validate_v16(ref,item.copy(),series,"b"*64)
    altered=item.copy()
    altered.loc[0,"taker_imbalance"]=0.8
    with pytest.raises(AssertionError,match="taker_imbalance"):
        validate_v16(ref,altered,series,"a"*64)
