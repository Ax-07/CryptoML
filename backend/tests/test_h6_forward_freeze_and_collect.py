"""Safety tests for isolated H6 freeze and one-shot prospective capture."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.h6_forward_freeze import make_frozen_models, predeclared_last_fold
from mlpipeline.h6_forward_collect_once import prepare_latest_shared_features
from mlpipeline.h6_forward_candidate_evaluation import START, SYMBOLS


def test_freeze_rejects_late_clock_without_reading_data():
    with pytest.raises(ValueError,match="before"):
        make_frozen_models({},now=START)


def test_freeze_rejects_unseen_symbol():
    with pytest.raises(ValueError,match="four seen"):
        predeclared_last_fold({"ADAUSDT":pd.DataFrame()})


def test_collector_rejects_empty_snapshot():
    with pytest.raises(ValueError,match="Missing OHLCV"):
        prepare_latest_shared_features(pd.DataFrame(),
                                       as_of_utc=START+pd.Timedelta(minutes=5))


def synthetic_bars():
    base=START-pd.Timedelta(minutes=100*5)
    result=[]
    for symbol in SYMBOLS:
        for i in range(101):
            op=base+pd.Timedelta(minutes=5*i)
            result.append({
                "symbol":symbol,
                "bar_open_at":op.isoformat(),
                "bar_close_at":(op+pd.Timedelta(minutes=5)).isoformat(),
                "open":100+i*.1,"high":101+i*.1,"low":99+i*.1,
                "close":100+i*.1,"volume_base":10.,"volume_quote":1000.,
                "trade_count":100,"taker_buy_volume_base":5.,
                "is_final":True,
            })
    return pd.DataFrame(result)


def test_collector_rejects_out_of_window_backfill_and_unclosed():
    bars=synthetic_bars()
    close=pd.to_datetime(bars.bar_close_at,utc=True).max()
    with pytest.raises(ValueError,match="Too old"):
        prepare_latest_shared_features(bars,as_of_utc=close+pd.Timedelta(minutes=3))
    with pytest.raises(ValueError,match="Too old"):
        prepare_latest_shared_features(bars,as_of_utc=close-pd.Timedelta(seconds=1))


def test_collector_rejects_nonfinal_and_incomplete_shared_symbols():
    bars=synthetic_bars()
    close=pd.to_datetime(bars.bar_close_at,utc=True).max()
    bars.loc[0,"is_final"]=False
    with pytest.raises(ValueError,match="final"):
        prepare_latest_shared_features(bars,as_of_utc=close+pd.Timedelta(seconds=15))
    bars=synthetic_bars()
    mask=(bars.symbol=="XRPUSDT")&(bars.bar_close_at==close.isoformat())
    bars=bars.loc[~mask]
    with pytest.raises(ValueError,match="Shared close mismatch"):
        prepare_latest_shared_features(bars,as_of_utc=close+pd.Timedelta(seconds=15))


def test_collector_accepts_causal_closed_finite_features():
    bars=synthetic_bars()
    close=pd.to_datetime(bars.bar_close_at,utc=True).max()
    t,rows=prepare_latest_shared_features(
        bars,as_of_utc=close+pd.Timedelta(seconds=15))
    assert t==close
    assert len(rows)==4
    assert all(len(frame)==1 for frame in rows)
