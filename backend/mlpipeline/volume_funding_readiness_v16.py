"""V16 read-only A06 taker/volume candidate census and optional public funding availability probe."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint, REQUIRED
from mlpipeline.pullback_long_history_v3 import SYMBOLS, FIRST_MONTH, LAST_MONTH, CUTOFF
from mlpipeline.multistrategy_census_v13 import _features as _unused  # no behavior dependency
from mlpipeline.trend_pullback import M5, _validate_closed_m5, _segment_features

PROTOCOL = "p4_v16_a06_taker_volume_unlabeled_v1"
FIELDS = ("symbol", "side", "decision_close_at", "reference_close", "atr14_m15",
          "taker_imbalance", "relative_volume")
M15 = pd.Timedelta(minutes=15)


def _load_full_p0(store:CandleStore):
    """Reuse immutable SHA + quality checks, then load additional archived raw volume fields."""
    from mlpipeline.pullback_long_history_v3 import load_long_development_history
    _, manifests = load_long_development_history(store)
    out = {}
    for symbol in SYMBOLS:
        frames = []
        for m in [x for x in manifests if x["symbol"] == symbol]:
            year, month = (int(v) for v in m["period"].split("-"))
            path = store.partition_dir(symbol,year,month)/"part-000.parquet"
            f = pd.read_parquet(path,columns=list(REQUIRED))
            for col in ("bar_open_at","bar_close_at"):
                f[col]=pd.to_datetime(f[col],utc=True)
            f = f.loc[f.bar_close_at.lt(CUTOFF)].copy()
            f["symbol"]=symbol
            frames.append(f)
        out[symbol] = pd.concat(frames,ignore_index=True).sort_values("bar_open_at").reset_index(drop=True)
    return out, manifests


def _validate_volume(f:pd.DataFrame):
    _validate_closed_m5(f)
    cols=("volume_base","volume_quote","taker_buy_volume_base","taker_buy_volume_quote","trade_count")
    for c in cols:
        arr=pd.to_numeric(f[c],errors="raise").to_numpy(dtype=float)
        if not np.isfinite(arr).all() or (arr<0).any():
            raise ValueError(f"Non-finite or negative {c}")
    for total,taker in (("volume_base","taker_buy_volume_base"),
                        ("volume_quote","taker_buy_volume_quote")):
        if (f[taker].to_numpy(dtype=float)>f[total].to_numpy(dtype=float) +
                np.maximum(1e-8,1e-8*f[total].to_numpy(dtype=float))).any():
            raise ValueError(f"Taker volume exceeds total: {taker}")
    count=f.trade_count.to_numpy(dtype=float)
    if (count!=np.floor(count)).any():raise ValueError("Non-integral trade count")


def _aggregate_m15(f:pd.DataFrame)->pd.DataFrame:
    size=15*60*1_000_000_000
    stamp=f.bar_close_at.astype("int64").to_numpy()
    f=f.copy()
    f["_bucket"]=pd.to_datetime(((stamp-1)//size+1)*size,utc=True)
    rows=[]
    for close_at,g in f.groupby("_bucket",sort=True):
        if (len(g)!=3 or g.bar_close_at.iloc[-1]!=close_at or
            not g.bar_open_at.diff().iloc[1:].eq(M5).all()):
            continue
        rows.append({
            "symbol":g.symbol.iloc[0],"bar_open_at":g.bar_open_at.iloc[0],
            "bar_close_at":close_at,"open":float(g.open.iloc[0]),
            "high":float(g.high.max()),"low":float(g.low.min()),
            "close":float(g.close.iloc[-1]),"volume_base":float(g.volume_base.sum()),
            "taker_buy_volume_base":float(g.taker_buy_volume_base.sum())})
    return pd.DataFrame(rows,columns=["symbol","bar_open_at","bar_close_at",
                                     "open","high","low","close",
                                     "volume_base","taker_buy_volume_base"])


def census_symbol(f:pd.DataFrame):
    _validate_volume(f)
    m=_aggregate_m15(f)
    out=[]
    if m.empty:return pd.DataFrame(columns=FIELDS),{"m5":len(f),"m15_complete":0,"m15_zero_volume":0}
    m["segment"]=m.bar_open_at.diff().ne(M15).cumsum()
    for _,segment in m.groupby("segment",sort=False):
        seg=_segment_features(segment.reset_index(drop=True))
        previous_median=seg.volume_base.shift(1).rolling(32,min_periods=32).median()
        last=-6
        for i,r in enumerate(seg.itertuples(index=False)):
            volume=float(r.volume_base)
            if (i-last<6 or volume<=0 or not np.isfinite(r.atr14) or
                r.atr14<=0 or not np.isfinite(previous_median.iloc[i]) or
                previous_median.iloc[i]<=0):
                continue
            ratio=volume/float(previous_median.iloc[i])
            imbalance=2*float(r.taker_buy_volume_base)/volume-1
            side=None
            if ratio>=1.5 and r.close-r.open>=0.35*r.atr14 and imbalance>=0.20:
                side="LONG"
            elif ratio>=1.5 and r.open-r.close>=0.35*r.atr14 and imbalance<=-0.20:
                side="SHORT"
            if side is not None:
                last=i
                out.append((r.symbol,side,r.bar_close_at,float(r.close),
                            float(r.atr14),imbalance,ratio))
    signals=pd.DataFrame(out,columns=FIELDS)
    if not signals.empty and signals.duplicated(["symbol","decision_close_at"]).any():
        raise AssertionError("Repeated signal")
    return signals,{"m5":len(f),"m15_complete":len(m),
                    "m15_zero_volume":int(m.volume_base.eq(0).sum()),
                    "long":int(signals.side.eq("LONG").sum()),
                    "short":int(signals.side.eq("SHORT").sum())}


def funding_probe(symbol:str):
    if symbol not in SYMBOLS:raise ValueError("Unapproved symbol")
    start=int(pd.Timestamp("2025-06-01T00:00:00Z").timestamp()*1000)
    end=int(pd.Timestamp("2025-06-03T00:00:00Z").timestamp()*1000)
    url="https://fapi.binance.com/fapi/v1/fundingRate?"+urlencode(
        {"symbol":symbol,"startTime":start,"endTime":end,"limit":100})
    with urlopen(Request(url,headers={"User-Agent":"CryptoML-V16-research-audit"}),timeout=12) as resp:
        if resp.status!=200:raise ValueError(f"HTTP {resp.status}")
        rows=json.loads(resp.read(300000))
    if not isinstance(rows,list):raise ValueError("Unexpected funding response")
    timestamps=[]
    for row in rows:
        if not isinstance(row,dict) or row.get("symbol")!=symbol:raise ValueError("Funding schema mismatch")
        timestamp=int(row["fundingTime"])
        rate=float(row["fundingRate"])
        if timestamp<start or timestamp>end or not np.isfinite(rate):
            raise ValueError("Funding timestamp/rate invalid")
        timestamps.append(timestamp)
    if timestamps!=sorted(set(timestamps)):raise ValueError("Funding timestamps not unique/sorted")
    return {"symbol":symbol,"records":len(rows),"window":"2025-06-01 through 2025-06-03 UTC",
            "first_ms":timestamps[0] if timestamps else None,
            "last_ms":timestamps[-1] if timestamps else None,
            "coverage_of_full_P0":"NOT_ESTABLISHED",
            "status":"HISTORICAL_SAMPLE_RETRIEVED" if rows else "NO_SAMPLE_FOUND"}


def main():
    parser=argparse.ArgumentParser(description="V16 A06 unlabeled, offline by default; funding probe optional")
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--output-json",required=True,type=Path)
    parser.add_argument("--output-csv",required=True,type=Path)
    parser.add_argument("--probe-funding-history",action="store_true")
    a=parser.parse_args()
    root=a.data_root.resolve()
    outputs=(a.output_json.resolve(),a.output_csv.resolve())
    if (outputs[0]==outputs[1] or any(p.exists() or p.is_relative_to(root)
        or not p.parent.is_dir() for p in outputs)):
        parser.error("Require two new output files outside P0")
    candles,manifests=_load_full_p0(CandleStore(root))
    frames=[];metadata={}
    for symbol in SYMBOLS:
        print("V16 verifying existing volume fields",symbol,flush=True)
        signals,stats=census_symbol(candles[symbol])
        frames.append(signals);metadata[symbol]=stats
    combined=pd.concat(frames,ignore_index=True)
    probe={}
    if a.probe_funding_history:
        for sym in SYMBOLS:
            try:probe[sym]=funding_probe(sym)
            except (OSError,ValueError,KeyError,TypeError,TimeoutError) as exc:
                probe[sym]={"symbol":sym,"status":"UNVERIFIED",
                            "error_type":type(exc).__name__,
                            "coverage_of_full_P0":"NOT_ESTABLISHED"}
    report={"protocol":PROTOCOL,"mode":"HISTORICAL_UNLABELED_NO_PNL",
            "fingerprint_sha256":manifest_fingerprint(manifests),
            "source_partitions":len(manifests),"symbols":list(SYMBOLS),
            "by_symbol":metadata,"total_signals":len(combined),
            "funding_probe":probe,"no_order_execution":True,
            "limitations":"Taker-volume signal is not an order-flow fill or proven edge. Funding probe never proves historical completeness. No future OOS."}
    with outputs[1].open("x",encoding="utf-8",newline="") as out:combined.to_csv(out,index=False)
    with outputs[0].open("x",encoding="utf-8") as out:json.dump(report,out,indent=2,allow_nan=False)
    print("A06", {s:(v["long"],v["short"]) for s,v in metadata.items()},flush=True)
    print("NO PNL / NO TRAINING / NO ORDERS",flush=True)


if __name__=="__main__":main()
