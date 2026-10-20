"""No P0 data access: strictly synthetic tests for P4 ML selector."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.pullback_ml import (
    _metrics,_part,confirmed_features,train_and_report,FEATURES,
    START,TRAIN_END,CALIB_END,MAY_END,CUTOFF,SYMBOLS,
)
from mlpipeline.pullback_baselines import annotate_confirmation

def test_features_only_use_bars_up_to_confirmation():
    dates=pd.date_range("2026-02-01T00:05:00Z",periods=18,freq="5min")
    bars=pd.DataFrame({"bar_open_at":dates-pd.Timedelta(minutes=5),"bar_close_at":dates,
                       "open":np.full(18,100.),"close":np.full(18,100.),
                       "high":np.full(18,101.),"low":np.full(18,99.)})
    bars.loc[4,"close"]=102.;bars.loc[5,"open"]=100.;bars.loc[5,"close"]=103.
    detected=pd.DataFrame({"bar_close_at":dates,"ema9":np.full(18,99.),
                           "ema26":np.full(18,98.),"atr14":np.full(18,2.),
                           "pullback_signal":[i==4 for i in range(18)]})
    outcomes=pd.DataFrame({"signal_close_at":[dates[4]],"entry_at":[dates[4]+pd.Timedelta(minutes=5)],
      "mature_at":[dates[4]+pd.Timedelta(minutes=35)],"gross_bps":[2.],
      "net_2bps":[0.],"net_5bps":[-3.],"net_10bps":[-8.],"exit_reason":["timeout"]})
    event=annotate_confirmation(bars,detected,outcomes)
    first=confirmed_features(bars,detected,event)
    assert bool(first.baseline_b_confirmed.iloc[0])
    assert np.isfinite(first.loc[0,list(FEATURES)].to_numpy(dtype=float)).all()
    bars.loc[7:,"close"]=9999.
    bars.loc[7:,"high"]=10000.
    bars.loc[7:,"low"]=1.
    later=confirmed_features(bars,detected,event)
    np.testing.assert_array_equal(first.loc[0,list(FEATURES)].to_numpy(dtype=float),
                                  later.loc[0,list(FEATURES)].to_numpy(dtype=float))

def test_purging_excludes_labels_maturing_after_partition_end():
    x=pd.DataFrame({"signal_close_at":[TRAIN_END-pd.Timedelta(minutes=5),TRAIN_END-pd.Timedelta(minutes=40)],
       "mature_at":[TRAIN_END+pd.Timedelta(minutes=30),TRAIN_END-pd.Timedelta(minutes=5)]})
    assert len(_part(x,START,TRAIN_END))==1

def test_metrics_zero_trade_matches_no_trade():
    x=pd.DataFrame({"net_2bps":[-4.],"net_5bps":[-7.],"net_10bps":[-12.],"exit_reason":["stop"]})
    result=_metrics(x,[False])
    assert result["net_5bps_per_opportunity"]==0.
    assert result["trades"]==0

def test_train_and_report_smoke_synthetic():
    # All synthetic labels, 20 events per month per symbol; no access to market data.
    ts=(list(pd.date_range("2026-02-10",periods=30,freq="1h",tz="UTC"))+
        list(pd.date_range("2026-04-10",periods=10,freq="1h",tz="UTC"))+
        list(pd.date_range("2026-05-10",periods=5,freq="1h",tz="UTC"))+
        list(pd.date_range("2026-06-10",periods=5,freq="1h",tz="UTC")))
    frames={}
    for k,symbol in enumerate(SYMBOLS):
        times=pd.DatetimeIndex(ts)
        f=pd.DataFrame({"symbol":symbol,"signal_close_at":times,
                        "mature_at":times+pd.Timedelta(minutes=35),
                        "baseline_b_confirmed":True,
                        "net_2bps":np.full(len(times),3.),
                        "net_5bps":np.full(len(times),0.),
                        "net_10bps":np.full(len(times),-5.),
                        "exit_reason":"timeout"})
        for i,key in enumerate(FEATURES):
            f[key]=1.+i*0.01+np.arange(len(f))*0.001+k*0.001
        frames[symbol]=f
    report,table=train_and_report(frames,"a"*64)
    assert report["train_B_events"]==120
    assert report["april_B_calibration_events"]==40
    assert len(table)==32
    assert all(row["trades"]==0 for month in report["diagnostics"].values()
               for symbol in month.values() for strategy,row in symbol.items() if strategy=="D_no_trade")

def test_only_four_approved_symbols():
    with pytest.raises(ValueError,match="Four approved"):
        train_and_report({},"a"*64)
