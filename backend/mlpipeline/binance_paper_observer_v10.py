"""One-shot, read-only Binance USD-M M5 collector + local paper signal journal.

No credentials, no trading endpoints, no background loop. Offline reconstructed
signals are not claimed to have been emitted live.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from mlpipeline.structure_breakout_retest_v9 import detect, SYMBOLS, PROTOCOL, PROSPECTIVE_START

API="https://fapi.binance.com"
PATH="/fapi/v1/klines"
M5_MS=300000
ROWS=1500
REQUIRED_WARMUP=600
SCHEMA="""CREATE TABLE IF NOT EXISTS observed_signals (
 symbol TEXT NOT NULL, side TEXT NOT NULL,
 breakout_close_at TEXT NOT NULL, decision_close_at TEXT NOT NULL,
 first_observed_utc TEXT NOT NULL, source_latest_close_at TEXT NOT NULL,
 level REAL NOT NULL, atr REAL NOT NULL, regime TEXT NOT NULL,
 source_sha256 TEXT NOT NULL, protocol TEXT NOT NULL,
 provenance TEXT NOT NULL,
 PRIMARY KEY(symbol,side,breakout_close_at,decision_close_at)
)"""


def request_klines(symbol:str,start_ms:int,end_ms:int,limit:int=ROWS):
    if symbol not in SYMBOLS or not 1<=limit<=1500 or end_ms<=start_ms:
        raise ValueError("Invalid Binance query bounds")
    url=API+PATH+"?"+urlencode({
        "symbol":symbol,"interval":"5m","startTime":start_ms,
        "endTime":end_ms,"limit":limit})
    with urlopen(Request(url,headers={"User-Agent":"CryptoML-paper-audit/1.0"}),timeout=25) as response:
        rows=json.load(response)
    if not isinstance(rows,list):
        raise ValueError("Exchange did not return candle array")
    return rows


def fetch_recent(symbol:str,now_ms:int,needed:int=REQUIRED_WARMUP):
    """Bounded paginated requests, ending before the current forming bar."""
    # One additional buffer allows rejecting the partial current close.
    latest_closed_end=(now_ms//M5_MS)*M5_MS
    end=latest_closed_end
    gathered={}
    for _ in range(5):
        begin=end-ROWS*M5_MS
        rows=request_klines(symbol,begin,end-1,ROWS)
        if not rows:
            break
        for r in rows:
            if len(r)<7:
                raise ValueError("Malformed Binance kline")
            opened=int(r[0]);closed=opened+M5_MS
            if closed<=latest_closed_end:
                gathered[opened]=r
        earliest=min(int(r[0]) for r in rows)
        if len(gathered)>=needed:
            break
        if earliest>=end:
            raise ValueError("Binance pagination did not move backward")
        end=earliest
    if len(gathered)<needed:
        raise ValueError(f"{symbol}: insufficient completed M5 history: {len(gathered)}")
    ordered=[gathered[t] for t in sorted(gathered)]
    df=pd.DataFrame({
        "symbol":symbol,
        "bar_open_at":pd.to_datetime([int(r[0]) for r in ordered],unit="ms",utc=True),
        "bar_close_at":pd.to_datetime([int(r[0])+M5_MS for r in ordered],unit="ms",utc=True),
        "open":[float(r[1]) for r in ordered],
        "high":[float(r[2]) for r in ordered],
        "low":[float(r[3]) for r in ordered],
        "close":[float(r[4]) for r in ordered],
        "is_final":True,
    })
    return df


def record(conn:sqlite3.Connection, signals:pd.DataFrame, observed_at:pd.Timestamp,
           candles:pd.DataFrame, symbol:str, freshness_mins:int=10):
    if observed_at.tzinfo is None or str(observed_at.tz)!="UTC":
        raise ValueError("Require UTC observation time")
    latest=candles.bar_close_at.iloc[-1]
    lag=observed_at-latest
    if lag<pd.Timedelta(0) or lag>pd.Timedelta(minutes=freshness_mins):
        raise ValueError(f"{symbol}: source candles not fresh; lag={lag}")
    digest=hashlib.sha256(candles[
        ["bar_open_at","open","high","low","close"]].to_csv(index=False).encode()).hexdigest()
    inserted=0
    # Only signals that have JUST been decided in the latest completed candle
    # can be called 'observed promptly'. Others are offline discoveries and
    # deliberately never stored as new observations in this journal.
    for event in signals.itertuples(index=False):
        if event.decision_close_at!=latest:
            continue
        if not (event.breakout_close_at>=PROSPECTIVE_START and
                event.breakout_close_at<event.decision_close_at<=observed_at):
            raise AssertionError("Prospective signal timing violation")
        result=conn.execute(
            """INSERT OR IGNORE INTO observed_signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol,event.side,event.breakout_close_at.isoformat(),
             event.decision_close_at.isoformat(),observed_at.isoformat(),
             latest.isoformat(),float(event.frozen_structure_level),
             float(event.atr_at_breakout),event.h1_regime,digest,PROTOCOL,
             "LOCAL_OBSERVATION_NOT_EXTERNAL_ATTESTATION"))
        inserted+=result.rowcount
    conn.commit()
    return inserted


def main():
    p=argparse.ArgumentParser(description="One-shot public Binance paper observer; no orders")
    p.add_argument("--journal",required=True,type=Path)
    a=p.parse_args()
    target=a.journal.resolve()
    if target.exists() and not target.is_file():
        p.error("Journal path must be a file")
    if not target.parent.is_dir() or target.suffix.lower()!=".sqlite3":
        p.error("Existing parent and .sqlite3 filename required")
    if pd.Timestamp.now(tz="UTC")<PROSPECTIVE_START:
        p.error("Prospective start not reached")
    datasets={}
    for symbol in SYMBOLS:
        began=time.time()
        now_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
        datasets[symbol]=fetch_recent(symbol,now_ms)
        print(symbol,"fetched",len(datasets[symbol]),"latest",datasets[symbol].bar_close_at.iloc[-1],
              "elapsed_s",round(time.time()-began,2),flush=True)
    # Read-only public endpoint; compute before opening journal.
    observations={sym:detect(datasets[sym]) for sym in SYMBOLS}
    observed_at=pd.Timestamp.now(tz="UTC")
    with sqlite3.connect(str(target)) as conn:
        conn.execute(SCHEMA)
        total=0
        for sym in SYMBOLS:
            count=record(conn,observations[sym],observed_at,datasets[sym],sym)
            print(sym,"new paper observations",count,flush=True)
            total+=count
    print("Saved",target,"new",total)
    print("LOCAL OBSERVATIONS ONLY: not exchange fills, no external clock attestation, no PnL")


if __name__=="__main__":
    main()
