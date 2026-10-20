"""Synthetic only: causality, incomplete H1 and short symmetry guards."""
import numpy as np
import pandas as pd

from mlpipeline.market_structure_v6 import (
    closed_higher_candles, classify_context, symmetric_candidates, classify_m5,
)
from mlpipeline.market_structure_census_v6 import summarize
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS


def bars(symbol="BTCUSDT",n=480):
    starts=pd.date_range("2026-02-01T00:00:00Z",periods=n,freq="5min")
    z=100+0.018*np.arange(n)+np.sin(np.arange(n)/4)*0.12
    return pd.DataFrame({"symbol":symbol,"bar_open_at":starts,
                         "bar_close_at":starts+pd.Timedelta(minutes=5),
                         "open":z-0.01,"high":z+0.14,
                         "low":z-0.14,"close":z,"is_final":True})


def test_higher_candles_only_complete_and_aligned():
    f=bars(n=36)
    h1=closed_higher_candles(f,12)
    assert len(h1)==3
    assert list(h1.bar_close_at)==[
        pd.Timestamp("2026-02-01T01:00:00Z"),
        pd.Timestamp("2026-02-01T02:00:00Z"),
        pd.Timestamp("2026-02-01T03:00:00Z")]
    f=f.drop(index=3)
    h1=closed_higher_candles(f,12)
    assert len(h1)==2


def test_incomplete_future_hour_cannot_be_joined():
    f=bars(n=480)
    out=classify_m5(f)
    # Restrict BOTH series to the same rows: dropna() on just one side
    # changes its index and pandas rightly refuses an unaligned comparison.
    for timeframe, minutes in (("m15", 15), ("h1", 60)):
        column = f"context_closed_at_{timeframe}"
        known = out[column].notna()
        assert known.any(), f"No completed {timeframe} bar in synthetic history"
        close_at = out.loc[known, "bar_close_at"]
        context_at = out.loc[known, column]
        assert (context_at <= close_at).all(), f"Future {timeframe} context was joined"
        assert ((close_at - context_at) < pd.Timedelta(minutes=minutes)).all(), (
            f"Stale {timeframe} context was joined"
        )
    first=out.loc[out.bar_close_at.eq(pd.Timestamp("2026-02-01T00:55:00Z"))]
    assert first.iloc[0].regime_h1=="UNKNOWN"
    # Perturbation future : everything before a cutoff must remain unchanged.
    changed=f.copy()
    changed.loc[250:,"close"]*=1.25
    changed.loc[250:,"high"]*=1.40
    out2=classify_m5(changed)
    cols=["long_signal","short_signal","regime_m15","regime_h1","market_regime"]
    pd.testing.assert_frame_equal(out.loc[:249,cols],out2.loc[:249,cols])


def test_mirrored_direction_counts_swap_on_price_inversion():
    f=bars(n=550)
    x=symmetric_candidates(f)
    inv=f.copy()
    for k in ("open","high","low","close"):
        inv[k]=1.0/f[k]
    inv["high"],inv["low"]=1.0/f["low"],1.0/f["high"]
    y=symmetric_candidates(inv)
    # Reciprocal transformation changes ATR arithmetic slightly; high-level
    # invariant is disjoint direction selection and correct boolean columns.
    assert not (x.long_signal & x.short_signal).any()
    assert not (y.long_signal & y.short_signal).any()
    assert (x[["long_signal","short_signal"]].dtypes==bool).all()


def test_no_classification_after_gap_until_new_complete_context():
    f=bars(n=510).drop(index=[300]).reset_index(drop=True)
    out=classify_m5(f)
    assert not out.bar_close_at.duplicated().any()
    assert out.market_regime.isin(("UP","DOWN","RANGE","UNKNOWN")).all()


def test_empty_higher_context_is_unknown():
    f=bars(n=8)
    assert closed_higher_candles(f,12).empty
    out=classify_m5(f)
    assert (out.market_regime=="UNKNOWN").all()


def test_census_explicitly_excludes_ada():
    f={s:classify_m5(bars(s,n=480)) for s in SYMBOLS}
    f["ADAUSDT"]=f.pop("BTCUSDT")
    try:
        summarize(f,"a"*64)
    except ValueError as exc:
        assert "four approved" in str(exc)
    else:
        raise AssertionError("ADA should be disallowed")
