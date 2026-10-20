"""Synthetic V10 journal tests: no web or P0 input."""
import sqlite3
import pandas as pd
import pytest
from mlpipeline.binance_paper_observer_v10 import SCHEMA,record,fetch_recent,M5_MS


def candles():
    start=pd.Timestamp("2026-10-22T00:00:00Z")
    return pd.DataFrame({
        "bar_open_at":[start,start+pd.Timedelta(minutes=5)],
        "bar_close_at":[start+pd.Timedelta(minutes=5),start+pd.Timedelta(minutes=10)],
        "open":[100.,101.],"high":[102.,103.],"low":[99.,100.],
        "close":[101.,102.]})

def signals():
    return pd.DataFrame([{
        "symbol":"BTCUSDT","side":"LONG",
        "breakout_close_at":pd.Timestamp("2026-10-22T00:05:00Z"),
        "decision_close_at":pd.Timestamp("2026-10-22T00:10:00Z"),
        "frozen_structure_level":100.,"atr_at_breakout":1.,
        "h1_regime":"UP","status":"OFFLINE_REPLAY_NOT_LIVE_ATTESTED"
    }])


def test_single_new_decision_and_idempotent_journal():
    c=candles();s=signals();now=pd.Timestamp("2026-10-22T00:11:00Z")
    with sqlite3.connect(":memory:") as conn:
        conn.execute(SCHEMA)
        assert record(conn,s,now,c,"BTCUSDT")==1
        assert record(conn,s,now,c,"BTCUSDT")==0
        assert conn.execute("select count(*) from observed_signals").fetchone()[0]==1
        assert conn.execute("select provenance from observed_signals").fetchone()[0]==(
            "LOCAL_OBSERVATION_NOT_EXTERNAL_ATTESTATION")


def test_stale_candles_rejected_even_with_signal():
    c=candles()
    with sqlite3.connect(":memory:") as conn:
        conn.execute(SCHEMA)
        with pytest.raises(ValueError,match="not fresh"):
            record(conn,signals(),pd.Timestamp("2026-10-22T00:30:00Z"),c,"BTCUSDT")


def test_old_signal_never_backfilled_as_new_observation():
    c=candles()
    older=signals().copy()
    older.loc[0,"decision_close_at"]=pd.Timestamp("2026-10-22T00:05:00Z")
    with sqlite3.connect(":memory:") as conn:
        conn.execute(SCHEMA)
        assert record(conn,older,pd.Timestamp("2026-10-22T00:11:00Z"),c,"BTCUSDT")==0


def test_fetch_rejects_insufficient_bars(monkeypatch):
    import mlpipeline.binance_paper_observer_v10 as module
    monkeypatch.setattr(module,"request_klines",lambda *args,**kwargs: [])
    with pytest.raises(ValueError,match="insufficient"):
        fetch_recent("BTCUSDT",int(pd.Timestamp("2026-10-22T04:00:00Z").timestamp()*1000))


def test_fetch_never_admits_forming_bar(monkeypatch):
    import mlpipeline.binance_paper_observer_v10 as module
    observed=[]
    end=int(pd.Timestamp("2026-10-22T04:02:00Z").timestamp()*1000)
    boundary=(end//M5_MS)*M5_MS
    opened=boundary-M5_MS
    def fake(symbol,start_ms,end_ms,limit):
        observed.append((start_ms,end_ms))
        # Deliberately return an out-of-range still-forming bar as well.
        # fetch_recent must NOT keep it, even if the provider misbehaves.
        return [[opened,100,101,99,100,0,0],
                [boundary,100,101,99,100,0,0]]
    monkeypatch.setattr(module,"request_klines",fake)
    complete=fetch_recent("BTCUSDT",end,needed=1)
    assert len(complete)==1
    assert complete.bar_open_at.iloc[0]==pd.Timestamp(opened,unit="ms",tz="UTC")
    assert complete.bar_close_at.iloc[-1]==pd.Timestamp(boundary,unit="ms",tz="UTC")
    assert observed
    assert all(b<boundary for a,b in observed)


def test_fetch_insufficient_after_valid_partial_page(monkeypatch):
    import mlpipeline.binance_paper_observer_v10 as module
    requests=[]
    end=int(pd.Timestamp("2026-10-22T04:02:00Z").timestamp()*1000)
    boundary=(end//M5_MS)*M5_MS
    opened=boundary-M5_MS

    def fake(symbol,start_ms,end_ms,limit):
        requests.append((start_ms,end_ms))
        # First page has only one complete candle. The next page is empty,
        # as opposed to repeating a row outside the requested older window.
        if len(requests)==1:
            return [[opened,100,101,99,100,0,0]]
        return []

    monkeypatch.setattr(module,"request_klines",fake)
    with pytest.raises(ValueError,match="insufficient"):
        fetch_recent("BTCUSDT",end,needed=2)
    assert len(requests)==2
    assert requests[1][1]<requests[0][1]
