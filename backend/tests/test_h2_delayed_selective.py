"""Synthetic P2 delayed H2 selective policy checks, no P0 or exchange."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.h2_delayed_selective import delayed_truth,policy,metrics,attach

def candles():
    return pd.DataFrame({
        "symbol":["BTCUSDT"]*6,
        "bar_open_at":pd.date_range("2025-01-01",periods=6,freq="5min",tz="UTC"),
        "bar_close_at":pd.date_range("2025-01-01 00:05",periods=6,freq="5min",tz="UTC"),
        "open":[100,101,102,103,104,105],
        "close":[101,102,103,104,105,106],
    })

def test_delayed_truth_uses_third_bar_not_next_or_current():
    x=candles()
    from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
    all_assets={sym:x.assign(symbol=sym) for sym in SEEN_SYMBOLS}
    out=delayed_truth(all_assets,pd.Timestamp("2025-01-01 00:35",tz="UTC"))
    btc=out.loc[out.symbol.eq("BTCUSDT")].reset_index(drop=True)
    assert btc.loc[0,"delayed_return_bps"]==pytest.approx((103/102-1)*10000)
    assert btc.loc[0,"delayed_mature_at"]==pd.Timestamp("2025-01-01 00:15",tz="UTC")
    assert btc.loc[4:,"delayed_return_bps"].isna().all()

def test_delayed_truth_rejects_gap_for_target():
    x=candles()
    x.loc[2:,"bar_open_at"]+=pd.Timedelta(minutes=5)
    x.loc[2:,"bar_close_at"]+=pd.Timedelta(minutes=5)
    from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
    out=delayed_truth({sym:x.assign(symbol=sym) for sym in SEEN_SYMBOLS},
                      pd.Timestamp("2025-01-01 00:45",tz="UTC"))
    assert pd.isna(out.loc[out.symbol.eq("BTCUSDT")].iloc[0].delayed_return_bps)

def test_fixed_abstention_threshold_and_zero_cost():
    sides=policy(np.array([-3.01,-3,0,3,3.01]))
    assert sides.tolist()==[-1,0,0,0,1]
    x=metrics(np.array([4.,2.,0.,-2.,5.]),sides)
    assert x["trades"]==2 and x["abstentions"]==3
    assert x["by_cost"]["2"]["net_bps_per_opportunity"]==pytest.approx((-4+5-4)/5)
    assert x["by_cost"]["5"]["net_bps_per_trade"]==pytest.approx((-4+5-10)/2)

def test_attach_no_source_key_loss():
    source=pd.DataFrame({"symbol":["BTCUSDT"],"bar_close_at":[pd.Timestamp("2025-01-01",tz="UTC")]})
    actual=source.assign(delayed_return_bps=[1.],delayed_mature_at=[
        pd.Timestamp("2025-01-01 00:10",tz="UTC")])
    assert attach(source,actual).delayed_return_bps.tolist()==[1.]
