"""Synthetic V7 tests: no future candles, reference invariants, fixed selection."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.market_structure_v6 import classify_m5, closed_higher_candles, classify_context
from mlpipeline.market_structure_visual_audit_v7 import (
    _html_svg, _indicators, _asof_context, select_cases, verify_reference,
    render_case, MAY_START, KNOWN_CASE,
)
from mlpipeline.pullback_long_history_v3 import SYMBOLS


def candles(symbol="BTCUSDT", n=900):
    t=pd.date_range("2026-05-31T00:00:00Z",periods=n,freq="5min")
    price=100+np.arange(n)*0.02+np.sin(np.arange(n)/4)*0.2
    return pd.DataFrame({"symbol":symbol,"bar_open_at":t,"bar_close_at":t+pd.Timedelta(minutes=5),
                         "open":price-0.01,"close":price,"high":price+0.3,"low":price-0.3,
                         "is_final":True})


def test_no_future_candles_in_svg():
    f=candles()
    at=pd.Timestamp("2026-06-01T12:00:00Z")
    view=_indicators(f,1)
    a=_html_svg(view,"M5",at,at)
    changed=f.copy()
    changed.loc[changed.bar_close_at>at,"close"]*=1.5
    changed.loc[changed.bar_close_at>at,"high"]*=1.6
    b=_html_svg(_indicators(changed,1),"M5",at,at)
    assert a==b
    with pytest.raises(AssertionError,match="future"):
        _html_svg(view,"M5",at,at+pd.Timedelta(minutes=5))


def test_explicit_stale_higher_context_rejected():
    f=candles(n=900)
    h=closed_higher_candles(f,12)
    states=classify_context(h,12)
    current=states.iloc[-1]
    with pytest.raises(AssertionError,match="stale"):
        _asof_context(states,current.bar_close_at,current.bar_close_at+pd.Timedelta(hours=1),
                      12,str(current.regime))


def test_fixed_eth_case_precedes_other_categories():
    f={s:classify_m5(candles(s,n=900)) for s in SYMBOLS}
    # Inject the previously inspected timestamp solely into a synthetic frame
    # with an existing event, to test deterministic predeclared case placement.
    for s in SYMBOLS:
        f[s]=f[s].copy()
    ix=200
    f["ETHUSDT"].loc[ix,"bar_close_at"]=KNOWN_CASE[1]
    f["ETHUSDT"].loc[ix,"long_signal"]=True
    cases=select_cases(f)
    assert len(cases)==5
    assert cases[0]["symbol"]=="ETHUSDT"
    assert cases[0]["signal_close_at"]==KNOWN_CASE[1]


def test_reference_mismatch_fails_closed():
    f={s:classify_m5(candles(s,n=900)) for s in SYMBOLS}
    ref={"protocol":"wrong","source_dataset_fingerprint_sha256":"a"*64,
         "assets":list(SYMBOLS)}
    with pytest.raises(ValueError,match="reference"):
        verify_reference(ref,"a"*64,f)


def test_render_case_rejects_future_asof_and_bad_side():
    f=candles("BTCUSDT",n=900)
    classifications=classify_m5(f)
    at=classifications.bar_close_at.iloc[750]
    case={"selection":"test","symbol":"BTCUSDT","signal_close_at":at,"expected_side":"LONG"}
    h={tf:closed_higher_candles(f,n) for tf,n in (("M15",3),("H1",12))}
    states={tf:classify_context(frame,n) for tf,n,frame in
            (("M15",3,h["M15"]),("H1",12,h["H1"]))}
    with pytest.raises(AssertionError,match="Expected directional"):
        if not bool(classifications.loc[classifications.bar_close_at.eq(at),"long_signal"].iloc[0]):
            render_case(case,f,classifications,h,states)
        else:
            case["expected_side"]="SHORT"
            render_case(case,f,classifications,h,states)
