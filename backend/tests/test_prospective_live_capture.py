"""Synthetic P3 live-capture safety tests. No network, no future labels."""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.prospective_protocol import register, activate
from mlpipeline.prospective_live_capture import (
    latest_common_features, append_one_batch,
)


def utc(day,hour=0,minute=0):
    return datetime(2026,10,day,hour,minute,tzinfo=timezone.utc)


def test_misaligned_closed_bars_are_rejected():
    n=80
    times=pd.date_range("2026-10-15T00:00:00Z",periods=n,freq="5min")
    def bars(symbol,offset=0):
        close=100+np.linspace(0,2,n)
        opened=np.r_[close[0],close[:-1]]
        return pd.DataFrame({
            "symbol":symbol,
            "bar_open_at":times+pd.Timedelta(minutes=offset),
            "bar_close_at":times+pd.Timedelta(minutes=5+offset),
            "open":opened,"high":close+0.2,"low":opened-0.2,"close":close,
            "volume_base":np.full(n,10.),"volume_quote":np.full(n,1000.),
            "trade_count":np.full(n,100),"taker_buy_volume_base":np.full(n,5.),
            "taker_buy_volume_quote":np.full(n,500.),
        })
    from mlpipeline.prospective_protocol import ASSETS
    samples={s:bars(s) for s in ASSETS}
    samples["XRPUSDT"]=bars("XRPUSDT",5)
    with pytest.raises(ValueError, match="timestamp-aligned"):
        latest_common_features(samples,now=utc(15,6,35))


def test_local_journal_requires_activation_and_future_time(tmp_path):
    root=tmp_path/"p3"
    register(root,now=utc(14))
    forecasts={
        h:{sym:{"raw_decision":1} for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
        for h in ("H2","H6")
    }
    with pytest.raises(FileNotFoundError):
        append_one_batch(root,close=utc(16,0,5),now=utc(16,0,6),forecasts=forecasts)
