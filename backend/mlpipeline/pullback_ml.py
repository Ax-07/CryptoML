"""P4 experimental single Ridge selector on confirmed EMA pullbacks.

No search and no independent OOS claim. Uses only completed M5 candles from
Dec 2025--Jun 2026, four seen symbols, with ALL July+ data untouched.
Predicts net return *after 5bp roundtrip assumed costs* at t+1 close, before
the predeclared hypothetical entry at open(t+2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlpipeline.pullback_frequency import CUTOFF, START, SYMBOLS
from mlpipeline.pullback_baselines import annotate_confirmation, _score
from mlpipeline.pullback_outcomes import research_outcomes
from mlpipeline.trend_pullback import M5, detect_pullbacks

PROTOCOL = "p4_confirmed_ema9_ridge_net5_fixed_v1"
FEATURES = (
    "ema_spread_atr", "close_above_ema9_atr", "atr_fraction",
    "ema9_slope_3_atr", "ema26_slope_3_atr",
    "confirmation_body_atr", "confirmation_range_atr",
    "confirmation_lower_wick_atr", "signal_depth_atr",
)
TRAIN_END = pd.Timestamp("2026-04-01T00:00:00Z")
CALIB_END = pd.Timestamp("2026-05-01T00:00:00Z")
MAY_END = pd.Timestamp("2026-06-01T00:00:00Z")
COST_BPS = 5
LABEL_CLIP_BPS = 100.0
RIDGE_ALPHA = 10.0
APRIL_KEEP_QUANTILE = 0.80


def confirmed_features(candles: pd.DataFrame, detected: pd.DataFrame,
                       events: pd.DataFrame) -> pd.DataFrame:
    """Build features of B candidates using bars <= t+1 ONLY.

    Entry open(t+2), future outcomes, exit prices and labels are never used in
    any feature. This function is safe to rerun with later candles changed.
    """
    if (len(candles) != len(detected)
            or not candles.bar_close_at.equals(detected.bar_close_at)
            or not candles.bar_close_at.is_unique):
        raise ValueError("Misaligned or duplicated detector/candles")
    result = events.copy()
    for name in FEATURES:
        result[name] = np.nan
    index_at = {stamp: i for i, stamp in enumerate(candles.bar_close_at)}
    for row_index, row in result.iterrows():
        if not bool(row["baseline_b_confirmed"]):
            continue
        i = index_at.get(row["signal_close_at"])
        if i is None or i < 3 or i+1 >= len(candles):
            raise ValueError("Invalid confirmed candidate")
        j = i+1  # B is known at the close of j; entry cannot precede j CLOSE
        if (candles.bar_close_at.iloc[j] != candles.bar_close_at.iloc[i]+M5
                or candles.bar_open_at.iloc[i+2] != candles.bar_close_at.iloc[j]
                or row["entry_at"] != candles.bar_open_at.iloc[i+2]):
            raise ValueError("Missing confirmation bar or incorrect future entry clock")
        bar = candles.iloc[j]
        signal = candles.iloc[i]
        x = detected.iloc[j]
        prev3 = detected.iloc[j-3]
        atr = float(x.atr14)
        atr_signal = float(detected.atr14.iloc[i])
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(atr_signal)
                and atr_signal > 0):
            raise ValueError("Missing causal ATR")
        values = (
            (float(x.ema9)-float(x.ema26))/atr,
            (float(bar.close)-float(x.ema9))/atr,
            atr/float(bar.close),
            (float(x.ema9)-float(prev3.ema9))/atr,
            (float(x.ema26)-float(prev3.ema26))/atr,
            (float(bar.close)-float(bar.open))/atr,
            (float(bar.high)-float(bar.low))/atr,
            (min(float(bar.open),float(bar.close))-float(bar.low))/atr,
            (float(signal.low)-float(detected.ema9.iloc[i]))/atr_signal,
        )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite causal feature")
        for name, value in zip(FEATURES, values):
            result.at[row_index, name] = value
    return result


def labeled_events(candles: pd.DataFrame) -> pd.DataFrame:
    """Load event outcomes for research, never use outcomes as input features."""
    if (candles.empty or candles.symbol.nunique()!=1
            or candles.symbol.iloc[0] not in SYMBOLS
            or not candles.bar_close_at.ge(START).all()
            or not candles.bar_close_at.lt(CUTOFF).all()):
        raise ValueError("Only approved Dec-Jun seen-symbol P0 input")
    detected = detect_pullbacks(candles)
    outcome = research_outcomes(candles,development_end=CUTOFF)
    annotated = annotate_confirmation(candles,detected,outcome)
    return confirmed_features(candles,detected,annotated)


def _part(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Purged chronological partition: label must mature inside partition."""
    if not start < end <= CUTOFF:
        raise ValueError("Partition outside historical development")
    return frame.loc[(frame.signal_close_at >= start)
                     & (frame.signal_close_at < end)
                     & (frame.mature_at < end)].copy()


def _metrics(cohort: pd.DataFrame, selected: pd.Series | np.ndarray) -> dict:
    chosen = np.asarray(selected,dtype=bool)
    if len(chosen)!=len(cohort):
        raise ValueError("Selection length mismatch")
    subset=cohort.iloc[np.flatnonzero(chosen)]
    return _score(subset,total_opportunities=len(cohort))


def train_and_report(events_by_asset: dict[str,pd.DataFrame],fingerprint: str) -> tuple[dict,pd.DataFrame]:
    if set(events_by_asset) != set(SYMBOLS) or len(fingerprint)!=64:
        raise ValueError("Four approved symbol frames and provenance are mandatory")
    chunks=[]
    for symbol in SYMBOLS:
        f=events_by_asset[symbol].copy()
        if not {"signal_close_at","mature_at","baseline_b_confirmed","net_5bps",
                *FEATURES}.issubset(f.columns):
            raise ValueError(f"Missing label/feature fields for {symbol}")
        if not f.empty and (not f.signal_close_at.is_monotonic_increasing
                            or not f.signal_close_at.is_unique
                            or not f.signal_close_at.ge(START).all()
                            or not f.signal_close_at.lt(CUTOFF).all()
                            or not f.mature_at.lt(CUTOFF).all()):
            raise ValueError(f"Unexpected historical timestamps for {symbol}")
        f["symbol"]=symbol
        chunks.append(f)
    all_events=pd.concat(chunks,ignore_index=True)
    train=_part(all_events,START,TRAIN_END)
    april=_part(all_events,TRAIN_END,CALIB_END)
    may=_part(all_events,CALIB_END,MAY_END)
    june=_part(all_events,MAY_END,CUTOFF)
    train_b=train.loc[train.baseline_b_confirmed]
    april_b=april.loc[april.baseline_b_confirmed]
    if len(train_b)<100 or len(april_b)<30:
        raise ValueError("Insufficient train/April confirmed events; no fitting")
    for frame in (train_b,april_b):
        matrix=frame.loc[:,FEATURES].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite B features in model fitting/calibration")
    X_train=train_b.loc[:,FEATURES].to_numpy(dtype=float)
    y_train=train_b.net_5bps.to_numpy(dtype=float)
    if not np.isfinite(y_train).all():
        raise ValueError("Nonfinite training labels")
    model=make_pipeline(StandardScaler(),Ridge(alpha=RIDGE_ALPHA))
    # Training-only clipping is a predeclared robustness measure, NOT tuned.
    model.fit(X_train,np.clip(y_train,-LABEL_CLIP_BPS,LABEL_CLIP_BPS))
    scores_april=model.predict(april_b.loc[:,FEATURES].to_numpy(dtype=float))
    threshold=max(0.0,float(np.quantile(scores_april,APRIL_KEEP_QUANTILE)))
    if not np.isfinite(threshold):
        raise ValueError("Invalid fixed selection threshold")
    report={
        "protocol":PROTOCOL,
        "status":"EXPLORATORY_REUSED_HISTORY_NOT_INDEPENDENT_OOS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "seen_symbols":list(SYMBOLS),
        "feature_columns":list(FEATURES),
        "signal_time":"close(t); ML decision close(t+1); hypothetical entry open(t+2)",
        "train":"2025-12-01 to 2026-04-01 exclusive; purged label maturity",
        "calibration":"2026-04-01 to 2026-05-01 exclusive; purged; score quantile only",
        "exploratory_diagnostics":["2026-05","2026-06"],
        "training_target":"B candidate theoretical net_5bps clipped to +/-100 for FIT ONLY",
        "model":"StandardScaler + Ridge(alpha=10), no search",
        "threshold_rule":"score > max(0, April 80th percentile), fixed once",
        "score_threshold_predicted_net_5bps":threshold,
        "train_B_events":len(train_b),
        "april_B_calibration_events":len(april_b),
        "training_target_clipped_count":int((np.abs(y_train)>LABEL_CLIP_BPS).sum()),
        "warning":"Diagnostic months and strategy family previously inspected. Neither holdout nor independent prospective performance. Open-price fills hypothetical. 0-trade baseline = 0.",
        "diagnostics":{},
    }
    rows=[]
    for name,events in (("2026-05",may),("2026-06",june)):
        period={}
        for symbol in SYMBOLS:
            group=events.loc[events.symbol.eq(symbol)].copy()
            flag_b=group.baseline_b_confirmed.to_numpy(dtype=bool)
            selected=np.zeros(len(group),dtype=bool)
            if flag_b.any():
                X=group.loc[group.baseline_b_confirmed,FEATURES].to_numpy(dtype=float)
                if not np.isfinite(X).all():
                    raise ValueError("Nonfinite evaluation feature")
                scores=model.predict(X)
                selected[np.flatnonzero(flag_b)]=scores>threshold
            results={
                "A_all":_metrics(group,np.ones(len(group),dtype=bool)),
                "B_confirmed":_metrics(group,flag_b),
                "C_ridge_abstention":_metrics(group,selected),
                "D_no_trade":{"trades":0,"net_2bps_per_opportunity":0.0,
                              "net_5bps_per_opportunity":0.0,
                              "net_10bps_per_opportunity":0.0},
            }
            period[symbol]=results
            for key,val in results.items():
                rows.append({"month":name,"symbol":symbol,"strategy":key,
                             "opportunities":len(group),"trades":val["trades"],
                             "net_2bps_per_opportunity":val["net_2bps_per_opportunity"],
                             "net_5bps_per_opportunity":val["net_5bps_per_opportunity"],
                             "net_10bps_per_opportunity":val["net_10bps_per_opportunity"],
                             "net_5bps_per_trade":val.get("net_5bps_per_trade")})
        report["diagnostics"][name]=period
    return report,pd.DataFrame(rows)


def main()->None:
    p=argparse.ArgumentParser(description="P4 fixed Ridge selector exploratory (NO independent OOS)")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-csv",type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve()
    jout,cout=a.output_json.resolve(),a.output_csv.resolve()
    if (jout==cout or jout.parent!=cout.parent or not jout.parent.is_dir()
            or any(x.exists() or x.is_relative_to(root) for x in (jout,cout))):
        p.error("Outputs must be two new files in one existing directory outside P0")
    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint
    from mlpipeline.prospective_model_pack import load_clean_historic
    candles,manifests=load_clean_historic(CandleStore(root),verify_hashes=True)
    events={symbol:labeled_events(candles[symbol]) for symbol in SYMBOLS}
    report,table=train_and_report(events,manifest_fingerprint(manifests))
    with cout.open("x",encoding="utf-8",newline="") as f:
        table.to_csv(f,index=False)
    with jout.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print("Fixed threshold for predicted net@5bp:",report["score_threshold_predicted_net_5bps"])
    for month in ("2026-05","2026-06"):
        for symbol in SYMBOLS:
            entry=report["diagnostics"][month][symbol]
            b=entry["B_confirmed"];c=entry["C_ridge_abstention"]
            print(f"{month} {symbol}: B={b['trades']} C={c['trades']} "
                  f"net5/opp B={b['net_5bps_per_opportunity']:.3f} "
                  f"C={c['net_5bps_per_opportunity']:.3f}")
    print("EXPLORATORY ONLY; NO INDEPENDENT VALIDATION; NEVER-TRADE BASELINE=0")
    print("Saved",jout,cout)


if __name__=="__main__":
    main()
