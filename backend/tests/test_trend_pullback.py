"""Synthetic-only tests; do not access real P0, P2, holdouts or P3."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.trend_pullback import M5,detect_pullbacks
from mlpipeline.pullback_outcomes import research_outcomes

def candles(n=120):
    end=pd.date_range("2025-01-01T00:05:00Z",periods=n,freq="5min")
    close=100.0+0.12*np.arange(n)
    opening=close-0.04
    return pd.DataFrame({"symbol":"BTCUSDT","bar_open_at":end-M5,
        "bar_close_at":end,"open":opening,"high":close+0.09,
        "low":opening-0.04,"close":close,"is_final":True})

def with_pullback():
    bars=candles();t=80
    bars.loc[t,"open"]=bars.loc[t-1,"close"]-0.02
    bars.loc[t,"close"]=bars.loc[t-1,"close"]-0.33
    bars.loc[t,"low"]=bars.loc[t,"close"]-0.12
    bars.loc[t,"high"]=bars.loc[t,"open"]+0.04
    return bars,t

def test_first_touch_signal_after_close():
    bars,t=with_pullback();out=detect_pullbacks(bars)
    assert out.loc[t,"uptrend_prior"] and out.loc[t,"first_ema9_touch"]
    assert out.loc[t,"pullback_signal"]
    assert not out.loc[t-1,"pullback_signal"]
    assert out.loc[t,"bar_close_at"]==bars.loc[t,"bar_close_at"]
    assert not any("future" in x or "target" in x for x in out.columns)

def test_future_ohlc_mutations_never_change_previous_features_or_signals():
    bars,t=with_pullback()
    before=detect_pullbacks(bars).iloc[:t+1].reset_index(drop=True)
    changed=bars.copy()
    changed.loc[t+1:,["open","close","low","high"]]*=1.7
    after=detect_pullbacks(changed).iloc[:t+1].reset_index(drop=True)
    pd.testing.assert_frame_equal(before,after)

def test_missing_candle_resets_ema():
    bars,t=with_pullback()
    bars.loc[t:,"bar_open_at"]+=M5;bars.loc[t:,"bar_close_at"]+=M5
    out=detect_pullbacks(bars)
    assert out.loc[t,"segment_id"]!=out.loc[t-1,"segment_id"]
    assert pd.isna(out.loc[t,"ema9"])
    assert not out.loc[t,"pullback_signal"]

def test_reject_unclosed_bars_and_invalid_ohlc():
    bars,t=with_pullback();bars.loc[30,"is_final"]=False
    with pytest.raises(ValueError,match="closed five-minute"):
        detect_pullbacks(bars)
    bars,t=with_pullback();bars.loc[30,"low"]=bars.loc[30,"high"]+1
    with pytest.raises(ValueError,match="Invalid OHLC"):
        detect_pullbacks(bars)

def test_cutoff_discards_immature_outcome():
    bars,t=with_pullback()
    out=research_outcomes(bars,development_end=bars.loc[t,"bar_close_at"]+M5)
    assert out.empty

def test_delayed_entry_and_complete_horizon():
    bars,t=with_pullback()
    out=research_outcomes(bars,development_end=bars.bar_close_at.iloc[-1]+M5)
    item=out.loc[out.signal_close_at.eq(bars.loc[t,"bar_close_at"])].iloc[0]
    assert item.entry_at==bars.loc[t,"bar_close_at"]+M5
    assert item.mature_at==bars.loc[t,"bar_close_at"]+7*M5
    assert item.entry_price==bars.loc[t+2,"open"]
    assert item.net_10bps==pytest.approx(item.gross_bps-10)

def test_double_touch_conservatively_exits_at_stop():
    bars,t=with_pullback();i=t+2
    bars.loc[i,"high"]=bars.loc[i,"open"]+5
    bars.loc[i,"low"]=bars.loc[i,"open"]-5
    out=research_outcomes(bars,development_end=bars.bar_close_at.iloc[-1]+M5)
    item=out.loc[out.signal_close_at.eq(bars.loc[t,"bar_close_at"])].iloc[0]
    assert item.exit_reason=="both_hit_stop_first"
    assert item.exit_price<item.entry_price

def test_requires_utc_cutoff():
    with pytest.raises(ValueError,match="UTC"):
        research_outcomes(candles(),development_end=pd.Timestamp("2025-01-02"))
