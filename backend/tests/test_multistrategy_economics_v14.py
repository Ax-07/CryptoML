"""V14 synthetic regression tests. No P0 data, labels or network."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.multistrategy_economics_v14 import (
    STRATEGIES,HORIZONS,QUALIFY,score_symbol,stats,verify_signals
)
from mlpipeline.multistrategy_census_v13 import PROTOCOL as V13_PROTOCOL


def candles(n=800):
    t=pd.date_range("2026-05-01T00:00:00Z",periods=n,freq="5min")
    p=np.full(n,100.0)
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":t,
        "bar_close_at":t+pd.Timedelta(minutes=5),
        "open":p,"high":p+0.5,"low":p-0.5,"close":p,
        "is_final":True})


def event(bars,strategy,at=100,side="LONG",atr=1.):
    return pd.DataFrame([{"strategy":strategy,"symbol":"BTCUSDT",
        "side":side,"timeframe":"H1","decision_close_at":bars.bar_close_at.iloc[at],
        "reference_price":100.0,"atr_known":atr}])


@pytest.mark.parametrize("strategy",STRATEGIES)
@pytest.mark.parametrize("side",["LONG","SHORT"])
def test_each_family_matures_and_costs_are_fixed(strategy,side):
    b=candles()
    f,counts=score_symbol(b,event(b,strategy,side=side))
    assert counts[strategy]["matured"]==1
    assert f.iloc[0].horizon_M5==HORIZONS[strategy]
    assert np.isclose(f.iloc[0].gross_bps,0)
    assert np.isclose(f.iloc[0].net_2bps,-2)
    assert np.isclose(f.iloc[0].net_5bps,-5)
    assert np.isclose(f.iloc[0].net_10bps,-10)
    assert stats(f)["trades"]==1


def test_missing_future_excluded_no_made_up_bar():
    b=candles(800)
    b=b.drop(index=[102]).reset_index(drop=True)
    f,counts=score_symbol(b,event(b,"A01_MOMENTUM_H4",at=100))
    assert f.empty
    assert counts["A01_MOMENTUM_H4"]["missing_or_immature"]==1


def test_only_previous_known_signal_and_no_future_close():
    b=candles()
    e=event(b,"A05_V9_RETEST",at=788)
    f,counts=score_symbol(b,e)
    assert f.empty
    assert counts["A05_V9_RETEST"]["missing_or_immature"]==1


def test_bad_atr_excluded_without_score():
    b=candles()
    f,counts=score_symbol(b,event(b,"A04_MEAN_REVERSION_H1",atr=0.))
    assert f.empty
    assert counts["A04_MEAN_REVERSION_H1"]["invalid_atr_or_price"]==1


def test_reference_fingerprint_and_event_parity():
    b=candles()
    signal=event(b,"A02_DONCHIAN_H1")
    computed={"BTCUSDT":signal,"ETHUSDT":signal.iloc[:0].copy(),
              "SOLUSDT":signal.iloc[:0].copy(),"XRPUSDT":signal.iloc[:0].copy()}
    counts={s:{k:{side:0 for side in ("LONG","SHORT")} for k in STRATEGIES}
            for s in computed}
    counts["BTCUSDT"]["A02_DONCHIAN_H1"]["LONG"]=1
    reference={"protocol":V13_PROTOCOL,"fingerprint_sha256":"a"*64,
               "assets":list(computed),"total_signals":1,"by_asset":counts}
    verify_signals(reference,signal.copy(),computed,"a"*64)
    with pytest.raises(ValueError,match="identity"):
        verify_signals(reference,signal.copy(),computed,"b"*64)
    bad=signal.copy()
    bad.loc[0,"reference_price"]=999.
    with pytest.raises(AssertionError,match="reference_price"):
        verify_signals(reference,bad,computed,"a"*64)


def test_reject_fee_corruption():
    b=candles()
    f,_=score_symbol(b,event(b,"A02_DONCHIAN_H1"))
    f.loc[0,"net_5bps"]=100.
    with pytest.raises(AssertionError,match="fee"):
        stats(f)
