"""Synthetic H1 paired audit tests; no private data."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.h1_sign_reference_audit import scope_metrics, past_candles


def fixture():
    return pd.DataFrame({
        "next_bar_return":[.01,-.02,.03,0.],
        "previous_return":[-.01,.02,0.,-.02],
        "lightgbm_30_sign":[2,0,0,2],
        "lightgbm_ema2_sign":[0,0,2,2],
        "ema_cross_rule_sign":[0,2,0,0],
    })


def test_exact_paired_nonflat_and_tie_policy():
    x=scope_metrics(fixture())
    assert x["n"]==3 and x["flat"]==1
    assert x["up"]==2 and x["down"]==1
    assert x["previous_flat"]==1
    assert x["accuracy"]["lightgbm_30"]==pytest.approx(2/3)
    assert x["accuracy"]["lightgbm_ema2"]==pytest.approx(2/3)
    assert x["accuracy"]["always_down"]==pytest.approx(1/3)
    assert x["accuracy"]["last_momentum"]==pytest.approx(0.)
    assert x["accuracy"]["last_reversion"]==pytest.approx(1.)
    pair=x["paired"]["lightgbm_30_minus_lightgbm_ema2"]
    assert pair["a_only"]==1 and pair["b_only"]==1
    assert pair["points"]==pytest.approx(0.)


def test_only_complete_previous_candle_accepted():
    times=pd.date_range("2025-01-01",periods=6,freq="5min",tz="UTC")
    data=pd.DataFrame({
        "bar_open_at":times,
        "bar_close_at":times+pd.Timedelta(minutes=5),
        "close":[100,101,102,103,104,105],
    })
    data.loc[3:,"bar_open_at"]+=pd.Timedelta(minutes=5)
    data.loc[3:,"bar_close_at"]+=pd.Timedelta(minutes=5)
    result=past_candles({"BTCUSDT":data,"ETHUSDT":data,
                         "SOLUSDT":data,"XRPUSDT":data},
                        pd.Timestamp("2025-01-02",tz="UTC"))
    btc=result.loc[result.symbol.eq("BTCUSDT")].reset_index(drop=True)
    assert np.isnan(btc.loc[3,"previous_return"])
    assert np.isnan(btc.loc[2,"verified_next_return"])
    assert btc.loc[4,"previous_return"]==pytest.approx(104/103-1)


def test_no_data_no_metrics():
    assert scope_metrics(fixture().iloc[:0]) is None
