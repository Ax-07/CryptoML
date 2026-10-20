"""Synthetic only: immutable mechanics/cost audit tests; no real P0 access."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.pullback_economics_audit_v4 import _stats,economics_audit,REASONS
from mlpipeline.pullback_long_history_v3 import SYMBOLS


def sample(symbol):
    times=pd.date_range("2025-01-08T00:05:00Z",periods=12,freq="1h")
    gross=np.array([-12.,0.,18.,-6.,15.,-20.,5.,2.,-3.,8.,-8.,1.])
    return pd.DataFrame({
        "symbol":symbol,"signal_close_at":times,
        "mature_at":times+pd.Timedelta(minutes=35),
        "baseline_b_confirmed":[True,False]*6,
        "gross_bps":gross,"net_2bps":gross-2,
        "net_5bps":gross-5,"net_10bps":gross-10,
        "exit_reason":["stop","timeout","target","stop",
                       "target","gap_stop","timeout","timeout",
                       "both_hit_stop_first","target",
                       "stop","gap_target_capped"]
    })


def test_arithmetic_and_exit_reason_conservation():
    f=sample("BTCUSDT")
    s=_stats(f,len(f))
    assert s["trades"]==12
    assert s["sum_gross_bps"]==sum(f.gross_bps)
    assert s["sum_net_5bps"]==sum(f.gross_bps)-60
    assert sum(s["exit_reasons_fixed"].values())==12
    assert s["gross_bps_quantiles"]["p50"]==float(np.median(f.gross_bps))


def test_reject_corrupt_fee_or_reason():
    f=sample("BTCUSDT")
    f.loc[0,"net_5bps"]=1000.
    with pytest.raises(AssertionError,match="cost arithmetic"):
        _stats(f,len(f))
    f=sample("BTCUSDT")
    f.loc[0,"exit_reason"]="future_target"
    with pytest.raises(ValueError,match="Unknown exit"):
        _stats(f,len(f))


def test_empty_confirmed_cohort_is_explicit():
    f=sample("BTCUSDT").iloc[:0]
    a=_stats(f,3)
    assert a["trades"]==0
    assert a["gross_bps_mean"] is None
    assert a["net_5bps_per_opportunity"]==0.


def test_four_asset_month_reconciliation():
    frames={symbol:sample(symbol) for symbol in SYMBOLS}
    report,rows=economics_audit(frames,"a"*64)
    assert report["overall"]["A_all"]["trades"]==48
    assert report["overall"]["B_confirmed"]["trades"]==24
    assert len(rows)==8
    assert report["overall"]["B_confirmed"]["opportunities"]==48
    assert all(rows.opportunities==12)


def test_reject_ada_and_reserved_date():
    frames={symbol:sample(symbol) for symbol in SYMBOLS}
    frames["ADAUSDT"]=frames.pop("BTCUSDT")
    with pytest.raises(ValueError,match="Four approved"):
        economics_audit(frames,"a"*64)
    frames={symbol:sample(symbol) for symbol in SYMBOLS}
    frames["BTCUSDT"].loc[11,"mature_at"]=pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError,match="Out-of-scope"):
        economics_audit(frames,"a"*64)
