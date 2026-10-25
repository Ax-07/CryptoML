"""P4 V36: causal native M5 feature contract and *metadata-only* readiness audit.

Never loads the June 2026 internal blind period into feature generation,
never accesses the Jul-Sep external holdout, and never creates labels, trains,
backtests, calculates PnL, sends orders or exports price/feature rows.

V36 is a NEW feature schema, NOT a drop-in replacement for features_v1.
Funding rates are excluded until publication-time/as-of evidence is reviewed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_unified_dev_reader_v34 import (
    APPROVED_FREEZE_SHA256, MARKET_FIELDS, M5, UnifiedDevelopmentReader,
    sha_file,
)
from mlpipeline.funding_basis_walkforward_preflight_v35 import (
    PROTOCOL as V35_PROTOCOL, INTERNAL_RESERVE_START, make_calendar_folds,
)
from mlpipeline.pullback_long_history_v3 import (
    SYMBOLS, EXPECTED_MONTHS, FIRST_MONTH, CUTOFF,
)

PROTOCOL = "p4_v36_causal_native_features_readiness_v1"
FEATURE_SCHEMA_VERSION = "p4_native_closed_m5_features_v36_v1"
FEATURE_COLUMNS = (
    "spot_ret_1", "spot_ret_12",
    "perp_ret_1", "perp_ret_12",
    "mark_ret_1", "mark_ret_12",
    "spot_volatility_48", "perp_volatility_48", "mark_volatility_48",
    "spot_ema_gap_26", "perp_ema_gap_26", "mark_ema_gap_26",
    "spot_range_fraction", "perp_range_fraction", "mark_range_fraction",
    "perp_mark_basis", "perp_spot_basis",
    "spot_volume_rel_48", "perp_volume_rel_48",
)
LOOKBACK_BARS = 48
ELIGIBLE_MONTHS = EXPECTED_MONTHS[:-1]
EXPECTED_V35_FOLDS = 33
V35_POLICY = {
    "train_days": 120, "validation_days": 14,
    "test_days": 14, "step_days": 14,
    "label_horizon_bars": 6,
    "embargo_bars_after_validation_and_test_starts": 6,
    "purging": "label_maturity_strictly_before_next_window_boundary",
    "rolling_train": True,
    "internal_june_blind_reserved": True,
}


def validate_contract():
    if (len(SYMBOLS) != 4 or len(ELIGIBLE_MONTHS) != 20
            or ELIGIBLE_MONTHS[0] != "2024-10"
            or ELIGIBLE_MONTHS[-1] != "2026-05"
            or INTERNAL_RESERVE_START != pd.Timestamp("2026-06-01T00:00:00Z")
            or CUTOFF != pd.Timestamp("2026-07-01T00:00:00Z")
            or len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS))
            or len(FEATURE_COLUMNS) != 19
            or LOOKBACK_BARS != 48
            or "mark_volume" in MARKET_FIELDS):
        raise ValueError("Frozen V36 causal research contract changed")


def _finite_numeric(market: pd.DataFrame, col: str):
    x = pd.to_numeric(market[col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite original market series: "+col)
    return x


def build_causal_features(market: pd.DataFrame) -> pd.DataFrame:
    """Only inputs from closed synchronized native M5. No future or funding.

    Must receive ONE contiguous UTC segment wholly before the blind reserve.
    Does not fabricate missing volume or NaNs in rolling warm-up windows.
    """
    validate_contract()
    if not isinstance(market, pd.DataFrame) or list(market.columns) != list(MARKET_FIELDS):
        raise ValueError("V36 requires exact V34 native market schema")
    if market.empty:
        raise ValueError("Empty market input")
    opens = pd.to_datetime(market["bar_open_at"], utc=True, errors="raise")
    closes = pd.to_datetime(market["bar_close_at"], utc=True, errors="raise")
    if (opens.isna().any() or closes.isna().any()
            or closes.iloc[0] <= pd.Timestamp(FIRST_MONTH+"-01T00:00:00Z")
            or closes.iloc[-1] >= INTERNAL_RESERVE_START
            or not closes.eq(opens + M5).all()
            or opens.duplicated().any()
            or not opens.diff().iloc[1:].eq(M5).all()):
        raise ValueError("Noncausal/out-of-scope/incomplete M5 timestamps")
    raw = {col: _finite_numeric(market,col) for col in MARKET_FIELDS[2:]}
    for kind in ("spot", "perp", "mark"):
        op,hi,lo,cl=(raw[kind+"_"+c] for c in ("open","high","low","close"))
        if (np.any(np.minimum.reduce((op,hi,lo,cl)) <= 0)
                or np.any(hi < np.maximum.reduce((op,lo,cl)))
                or np.any(lo > np.minimum.reduce((op,hi,cl)))):
            raise ValueError("Invalid native OHLC geometry: "+kind)
    for kind in ("spot","perp"):
        if np.any(raw[kind+"_volume"] < 0):
            raise ValueError("Negative native base volume: "+kind)

    f = pd.DataFrame({"bar_close_at": closes.reset_index(drop=True)})
    for kind in ("spot","perp","mark"):
        close = pd.Series(raw[kind+"_close"])
        high = pd.Series(raw[kind+"_high"])
        low = pd.Series(raw[kind+"_low"])
        logret = np.log(close).diff()
        f[kind+"_ret_1"] = close.pct_change(periods=1,fill_method=None)
        f[kind+"_ret_12"] = close.pct_change(periods=12,fill_method=None)
        f[kind+"_volatility_48"] = logret.rolling(48,min_periods=48).std()
        ema = close.ewm(span=26,adjust=False,min_periods=26).mean()
        f[kind+"_ema_gap_26"] = close.div(ema).sub(1.0)
        f[kind+"_range_fraction"] = (high-low).div(close)
    perp = pd.Series(raw["perp_close"])
    mark = pd.Series(raw["mark_close"])
    spot = pd.Series(raw["spot_close"])
    f["perp_mark_basis"] = perp.div(mark).sub(1.0)
    f["perp_spot_basis"] = perp.div(spot).sub(1.0)
    for kind in ("spot","perp"):
        vol = pd.Series(raw[kind+"_volume"])
        denom = vol.rolling(48,min_periods=48).mean().replace(0,np.nan)
        f[kind+"_volume_rel_48"] = vol.div(denom)

    if set(f.columns) != {"bar_close_at", *FEATURE_COLUMNS}:
        raise AssertionError("V36 feature schema drift")
    f = f.loc[:,["bar_close_at",*FEATURE_COLUMNS]]
    # Keep NaN in warmup/zero-volume rows, but never allow infinities.
    if np.isinf(f[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).any():
        raise ValueError("Feature infinity, no implicit imputation permitted")
    return f


def verify_v35_report(path: Path, expected_sha: str) -> dict:
    if (not re.fullmatch(r"[0-9a-f]{64}",expected_sha)
            or not path.is_file() or sha_file(path)!=expected_sha):
        raise ValueError("V35 audit SHA256 missing or changed")
    d = json.loads(path.read_bytes())
    if (d.get("protocol")!=V35_PROTOCOL
            or d.get("source_provenance_sha256")!=APPROVED_FREEZE_SHA256
            or d.get("status")!="CALENDAR_PREFLIGHT_PASS_ML_PIPELINE_BLOCKED"
            or d.get("fold_count")!=EXPECTED_V35_FOLDS
            or d.get("policy")!=V35_POLICY
            or d.get("calendar",{}).get("internal_blind_start_utc")!=INTERNAL_RESERVE_START.isoformat()
            or d.get("calendar",{}).get("external_holdout_start_utc")!=CUTOFF.isoformat()
            or d.get("no_network_no_holdout_access_no_training_no_labels_no_pnl_no_orders") is not True):
        raise ValueError("V35 preregistered calendar contract mismatch")
    folds = make_calendar_folds()
    if (len(folds)!=EXPECTED_V35_FOLDS
            or any(pd.Timestamp(f["test_end_exclusive_utc"])>INTERNAL_RESERVE_START
                   for f in folds)):
        raise ValueError("V35 calendar drift")
    return {"sha256":expected_sha,"fold_count":len(folds)}


def run(freeze_path: Path, freeze_sha: str,
        v35_path: Path, v35_sha: str,
        output_json: Path, output_csv: Path):
    output_json,output_csv=output_json.resolve(),output_csv.resolve()
    if (output_json==output_csv or output_json.exists() or output_csv.exists()
            or not output_json.parent.is_dir() or not output_csv.parent.is_dir()):
        raise ValueError("Two distinct NEW metadata report paths required")
    if freeze_sha!=APPROVED_FREEZE_SHA256:
        raise ValueError("V33 frozen provenance not approved")
    validate_contract()
    v35=verify_v35_report(v35_path,v35_sha)
    reader=UnifiedDevelopmentReader.preflight(freeze_path,freeze_sha)
    roots=reader._roots
    for p in (output_json,output_csv):
        if p in (freeze_path.resolve(),v35_path.resolve()) or any(
            p==root or p.is_relative_to(root) for root in roots[:4]
        ):
            raise ValueError("Report would modify original data or provenance")
    records=[]
    expected_total=0
    for symbol in SYMBOLS:
        for month in ELIGIBLE_MONTHS:
            market,_funding,status=reader.read_month(symbol,month)
            if status not in ("VERIFIED_V21","VERIFIED_V23"):
                raise ValueError("May 2026 and prior cannot use V33 partitions")
            # May final bar closes *exactly* at the June internal blind start.
            # It is verified by V34 then excluded before feature computation.
            last_cut = month == ELIGIBLE_MONTHS[-1]
            if last_cut:
                if (market.bar_close_at.ge(INTERNAL_RESERVE_START).sum()!=1
                        or market.bar_close_at.iloc[-1]!=INTERNAL_RESERVE_START):
                    raise ValueError("Unexpected May/June internal cutoff")
                market=market.loc[market.bar_close_at.lt(INTERNAL_RESERVE_START)].copy()
            if market.empty or market.bar_close_at.ge(INTERNAL_RESERVE_START).any():
                raise ValueError("Internal June blind leaked into feature input")
            derived=build_causal_features(market)
            available=np.isfinite(derived[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
            n=int(len(derived))
            ready=int(available.sum())
            if n<LOOKBACK_BARS+1 or ready<1:
                raise ValueError("No complete causal feature rows for "+symbol+" "+month)
            expected_total+=n
            records.append({
                "symbol":symbol,"month":month,
                "source_status":status,
                "closed_native_m5_before_internal_blind":n,
                "causal_feature_complete_rows":ready,
                "ineligible_due_to_warmup_or_zero_volume":n-ready,
                "internal_blind_cutoff_bar_excluded":int(last_cut),
                "last_eligible_close_utc":derived.bar_close_at.iloc[-1].isoformat(),
            })
    if len(records)!=80 or sum(r["internal_blind_cutoff_bar_excluded"] for r in records)!=4:
        raise ValueError("V36 observed unexpected development-month inventory")
    by_symbol={}
    for symbol in SYMBOLS:
        subset=[r for r in records if r["symbol"]==symbol]
        if len(subset)!=20 or sum(r["closed_native_m5_before_internal_blind"] for r in subset)!=175103:
            raise ValueError("V36 development M5 count mismatch: "+symbol)
        by_symbol[symbol]={
            "native_m5_before_internal_blind":sum(r["closed_native_m5_before_internal_blind"] for r in subset),
            "all_features_complete_rows":sum(r["causal_feature_complete_rows"] for r in subset),
            "ineligible_rows":sum(r["ineligible_due_to_warmup_or_zero_volume"] for r in subset),
        }
    report={
        "protocol":PROTOCOL,
        "v33_freeze_sha256":freeze_sha,
        "v35_report_sha256":v35["sha256"],
        "frozen_folds":v35["fold_count"],
        "feature_schema_version":FEATURE_SCHEMA_VERSION,
        "feature_columns":list(FEATURE_COLUMNS),
        "feature_count":len(FEATURE_COLUMNS),
        "market_data_scope":"2024-10-through-2026-05-with-May-June-boundary-bar-excluded",
        "last_feature_input_close_exclusive_utc":INTERNAL_RESERVE_START.isoformat(),
        "source_partitions_used_for_feature_diagnostics":len(records),
        "by_symbol":by_symbol,
        "native_source_columns":list(MARKET_FIELDS),
        "funding_features_status":"EXCLUDED_NOT_ASOF_ATTESTED",
        "legacy_feature_schema_status":"NOT_COMPATIBLE_REQUIRES_EXPLICIT_NEW_MODEL_PROTOCOL",
        "feature_calculation":"PAST_AND_CURRENT_CLOSED_M5_ONLY_NO_FUTURE",
        "status":"CAUSAL_NATIVE_FEATURE_DIAGNOSTICS_PASS_FUNDING_AND_ML_BLOCKED",
        "no_network_no_holdout_access_no_labels_no_training_no_backtest_no_pnl_no_orders":True,
        "no_price_or_feature_row_export_no_imputation":True,
    }
    with output_csv.open("x",encoding="utf-8",newline="") as f:
        pd.DataFrame(records).to_csv(f,index=False)
    with output_json.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    return report


def main():
    p=argparse.ArgumentParser(description="V36 native closed-bar causal feature readiness, metadata ONLY")
    for name in ("freeze-json","freeze-sha256","v35-json","v35-sha256","output-json","output-csv"):
        p.add_argument("--"+name,required=True,
                       type=Path if name.endswith("json") and name in ("freeze-json","v35-json","output-json","output-csv") else str)
    a=p.parse_args()
    result=run(Path(a.freeze_json),a.freeze_sha256,Path(a.v35_json),a.v35_sha256,
               Path(a.output_json),Path(a.output_csv))
    print("V36",result["status"],"features",result["feature_count"],flush=True)
    print("NO JUN 2026 FEATURES / NO JUL-SEP OOS / NO LABELS / NO TRAINING / NO ORDERS",flush=True)


if __name__=="__main__":
    main()
