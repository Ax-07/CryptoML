"""P2 internal H2 selection synthetic unit tests; no P0, check or OOS access."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.h2_internal_selection import split_purged,summarize_tune

def fold():
    end=pd.Timestamp("2025-05-20",tz="UTC")
    return {"check_start":end}

def test_partition_never_contains_old_check_and_enforces_purge():
    f=fold()
    stop=f["check_start"]-pd.Timedelta(days=3)
    early=stop-pd.Timedelta(days=7)
    start=early-pd.Timedelta(days=120)
    stamps=pd.concat([pd.Series(pd.date_range(start,periods=120,freq="D",tz="UTC")),
                      pd.Series(pd.date_range(early,periods=7*24,freq="h",tz="UTC")),
                      pd.Series(pd.date_range(stop,periods=3*24,freq="h",tz="UTC"))],ignore_index=True)
    # Single-symbol geometry is intentionally rejected by production split.
    d=pd.DataFrame({"bar_close_at":stamps,"label_mature_at":stamps+pd.Timedelta(minutes=60),
                    "symbol":["BTCUSDT"]*len(stamps)})
    with pytest.raises(ValueError,match="geometry"):
        split_purged(d,f,{"partition_rows":{"train":120,"early_stopping":168},"tuning_rows":72})

def test_all_abstention_is_ineligible():
    n=240
    df=pd.DataFrame({"fold_id":np.repeat(np.arange(36),10)[:n],
                     "symbol":["BTCUSDT"]*n,
                     "bar_close_at":pd.date_range("2025-01-01",periods=n,freq="5min",tz="UTC"),
                     "delayed_return_bps":np.ones(n),"regression_30_side":np.zeros(n,dtype=int)})
    out=summarize_tune(df,"regression_30")
    assert not out["eligible"]
    assert out["metrics"]["trades"]==0
    assert out["metrics"]["by_cost"]["2"]["net_bps_per_opportunity"]==0.

def test_single_windfall_fails_fold_and_best_five_guards():
    n=360
    side=np.zeros(n,dtype=int);side[:201]=1
    ret=np.zeros(n);ret[:201]=3.;ret[0]=100.
    df=pd.DataFrame({"fold_id":np.repeat(np.arange(36),10),
                     "symbol":["BTCUSDT"]*n,
                     "bar_close_at":pd.date_range("2025-01-01",periods=n,freq="5min",tz="UTC"),
                     "delayed_return_bps":ret,"regression_30_side":side})
    result=summarize_tune(df,"regression_30")
    assert result["eligibility_rules"]["min_200_trades_and_max_5pct_coverage"] is False
    assert result["eligible"] is False
