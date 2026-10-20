"""P2 exploratory H2/H6 label diagnostics, NO FITS, on prior dev checks only.

Loads only four seen assets, geometrically clips raw candles before the last
development check end BEFORE computing features/labels. All counted rows must
match the exact pre-existing 36-fold H12-purged validation-check geometry.
Historical P2 checks have been inspected repeatedly; no independent proof.
No evaluation of ADA, fold OOS or final90 and no model/production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS, build_shared_examples
from mlpipeline.feature_reduction_diagnostics import PROTOCOL as SOURCE_PROTOCOL

PROTOCOL = "p2_h2_h6_label_stability_devcheck_v1"
FOLDS = 36
HORIZONS = (2, 6)
MULTIPLIERS = (0.25, 0.5, 0.75, 1.0)
BASELINE_MULTIPLIER = 0.5
H12 = pd.Timedelta(minutes=60)
M5 = pd.Timedelta(minutes=5)
LABELS = ("DOWN", "NEUTRAL", "UP")


def load_plan(source: dict) -> list[dict]:
    """Read authoritative *existing* check boundaries; never generate fresh folds."""
    if (source.get("protocol") != SOURCE_PROTOCOL
            or source.get("mode") not in ("individual", "group")
            or source.get("research_status") != "EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS"
            or source.get("folds_analyzed") != FOLDS):
        raise ValueError("Require complete original P2 exploratory ablation report")
    for key in ("ADA_loaded_or_scored", "fold_OOS_scored", "final_90_days_scored",
                "frozen_P3_models_modified"):
        if source.get(key) is not False:
            raise ValueError(f"Forbidden or absent P2 safety assertion: {key}")
    folds = source.get("folds")
    if not isinstance(folds, list) or len(folds) != FOLDS:
        raise ValueError("Invalid fold plan")
    required = ("train_start_utc", "validation_start_utc",
                "early_stop_end_exclusive_utc", "unused_tuning_end_exclusive_utc",
                "validation_check_end_exclusive_utc")
    plan = []
    for i, entry in enumerate(sorted(folds, key=lambda f: f["fold_id"])):
        if type(entry.get("fold_id")) is not int or entry["fold_id"] != i:
            raise ValueError("Invalid fold ids")
        bounds = []
        for key in required:
            raw = pd.Timestamp(entry[key])
            if raw.tzinfo is None:
                raise ValueError("Naive UTC fold boundary")
            bounds.append(raw.tz_convert("UTC"))
        train, val, stop, tune, check = bounds
        if not (train < val < stop < tune < check):
            raise ValueError("Nonchronological fold")
        if not (val-train == pd.Timedelta(days=120)
                and stop-val == pd.Timedelta(days=7)
                and tune-stop == pd.Timedelta(days=3)
                and check-tune == pd.Timedelta(days=4)):
            raise ValueError("Fold split changed")
        counts = entry.get("partition_rows")
        if not isinstance(counts, dict) or not isinstance(counts.get("validation_check"), int):
            raise ValueError("Missing original validation check row count")
        plan.append({
            "fold_id":i,
            "check_start":tune,
            "check_end":check,
            "expected_check_rows":counts["validation_check"],
        })
    if any(plan[i]["check_end"] <= plan[i-1]["check_end"]
           for i in range(1, len(plan))):
        raise ValueError("Nonincreasing fold chronology")
    return plan


def label_from_return(ret: np.ndarray, vol: np.ndarray, horizon: int,
                      multiplier: float) -> np.ndarray:
    """Exactly emulate add_labels' comparisons; equality stays NEUTRAL."""
    if horizon not in HORIZONS or multiplier not in MULTIPLIERS:
        raise ValueError("Unknown horizon or threshold")
    ret = np.asarray(ret, dtype=float)
    vol = np.asarray(vol, dtype=float)
    if (ret.ndim != 1 or vol.shape != ret.shape or
            not np.isfinite(ret).all() or not np.isfinite(vol).all() or
            (vol < 0).any()):
        raise ValueError("Missing, negative volatility or non-finite future return")
    threshold = multiplier * vol * np.sqrt(horizon)
    return np.where(ret > threshold, 2, np.where(ret < -threshold, 0, 1)).astype("int8")


def shared_with_future_returns(candles: dict[str, pd.DataFrame],
                               last_check_end: pd.Timestamp) -> pd.DataFrame:
    """Clip BEFORE label construction; no future prices past check end accessed."""
    if set(candles) != set(SEEN_SYMBOLS):
        raise ValueError("Exactly four seen symbols, ADA prohibited")
    if last_check_end.tzinfo is None:
        raise ValueError("Expected UTC-aware cutoff")
    clipped = {
        symbol: frame.loc[frame.bar_close_at.lt(last_check_end)].copy()
        for symbol, frame in candles.items()
    }
    if any(frame.empty for frame in clipped.values()):
        raise ValueError("Empty clipped history")
    # Canonical seven-horizon eligible-row intersection and target builders.
    shared = build_shared_examples(clipped)
    if not shared.label_mature_at.lt(last_check_end).all():
        raise ValueError("Some H12 labels mature after the check cutoff")
    frames = []
    for symbol in SEEN_SYMBOLS:
        full = clipped[symbol].sort_values("bar_open_at").reset_index(drop=True)
        if full.bar_close_at.duplicated().any():
            raise ValueError("Duplicated candle close")
        segments = full.bar_open_at.diff().ne(M5).cumsum()
        grouped = full.groupby(segments, sort=False)
        keys = shared.loc[shared.symbol.eq(symbol), "bar_close_at"]
        joined = pd.DataFrame(index=full.bar_close_at)
        for horizon in HORIZONS:
            future_close = grouped["close"].shift(-horizon)
            future_time = grouped["bar_close_at"].shift(-horizon)
            future = (future_close/full.close-1).where(
                future_time.eq(full.bar_close_at + horizon*M5))
            joined[f"future_return_h{horizon}"] = future.to_numpy(dtype=float)
        frame = shared.loc[shared.symbol.eq(symbol)].copy()
        for h in HORIZONS:
            values = joined[f"future_return_h{h}"].reindex(keys).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("H2/H6 future price unexpectedly unavailable on common rows")
            frame[f"future_return_h{h}"] = values
            canonical = label_from_return(
                values, frame.volatility_48.to_numpy(dtype=float), h, 0.5)
            if not np.array_equal(canonical, frame[f"target_h{h}"].to_numpy(dtype=int)):
                raise AssertionError("Alternative labeling does not reproduce canonical add_labels")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "bar_close_at"]).reset_index(drop=True)
    if result.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate shared observation")
    return result


def describe_labels(frame: pd.DataFrame, horizon: int) -> dict:
    if horizon not in HORIZONS or frame.empty:
        raise ValueError("Nonempty valid H2/H6 observations required")
    ret = frame[f"future_return_h{horizon}"].to_numpy(dtype=float)
    vol = frame.volatility_48.to_numpy(dtype=float)
    baseline = frame[f"target_h{horizon}"].to_numpy(dtype=int)
    if not np.array_equal(label_from_return(ret, vol, horizon, 0.5), baseline):
        raise ValueError("Canonical label mismatch")
    n = len(frame)
    counts = np.bincount(baseline, minlength=3)
    movement_count = int(counts[0] + counts[2])
    bound = BASELINE_MULTIPLIER * vol*np.sqrt(horizon)
    near_boundary = (bound > 0) & (np.abs(np.abs(ret)-bound) <= 0.1*bound)
    # Purely descriptive retrospective class distribution: NOT model accuracy.
    thresholds = {}
    for mult in MULTIPLIERS:
        new = label_from_return(ret, vol, horizon, mult)
        current = np.bincount(new, minlength=3)
        key = str(mult)
        thresholds[key] = {
            "class_fractions":{LABELS[k]:float(current[k]/n) for k in range(3)},
            "changed_vs_0_5_fraction":float(np.mean(new != baseline)),
            "direction_to_neutral_fraction_of_all_rows":float(
                np.mean((baseline != 1) & (new == 1))),
            "neutral_to_direction_fraction_of_all_rows":float(
                np.mean((baseline == 1) & (new != 1))),
        }
    abs_bps = np.abs(ret)*10000
    return {
        "observations":n,
        "counts":{LABELS[k]:int(counts[k]) for k in range(3)},
        "fractions":{LABELS[k]:float(counts[k]/n) for k in range(3)},
        "movement_fraction":movement_count/n,
        "conditional_majority_direction_fraction_among_known_movers":
            float(max(counts[0], counts[2])/movement_count)
            if movement_count else None,
        "reference_warning":"Oracle descriptive only; movers not known at inference",
        "absolute_future_return_bps":{
            "median":float(np.median(abs_bps)),
            "p75":float(np.quantile(abs_bps,0.75)),
            "p90":float(np.quantile(abs_bps,0.90)),
        },
        "median_label_barrier_bps":float(np.median(bound)*10000),
        "near_original_barrier_within_10pct_fraction":float(np.mean(near_boundary)),
        "nonpositive_volatility_count":int((vol <= 0).sum()),
        "thresholds":thresholds,
    }


def run_audit(candles:dict[str,pd.DataFrame], source:dict,
              *,fingerprint:str,max_folds:int|None=None) -> dict:
    plan = load_plan(source)
    if source.get("dataset_fingerprint_sha256") != fingerprint:
        raise ValueError("Dataset fingerprint differs from original P2 report")
    if max_folds is not None and (type(max_folds) is not int or max_folds < 1):
        raise ValueError("max_folds must be a positive integer")
    if max_folds is not None:
        plan = plan[:max_folds]
    # Full report may contain later samples but no such labels are ever built:
    # cutoff is the last INCLUDED development check, not entire history.
    cutoff = plan[-1]["check_end"]
    shared = shared_with_future_returns(candles, cutoff)
    rows=[]
    for fold in plan:
        subset = shared.loc[
            shared.bar_close_at.ge(fold["check_start"])
            & shared.bar_close_at.lt(fold["check_end"])
            & shared.label_mature_at.lt(fold["check_end"])
        ]
        if len(subset) != fold["expected_check_rows"]:
            raise ValueError(
                f"Fold {fold['fold_id']} check observations mismatch "
                f"{len(subset)} != {fold['expected_check_rows']}"
            )
        if not set(subset.symbol)==set(SEEN_SYMBOLS):
            raise ValueError("Four known symbols missing in dev check")
        by_horizon={}
        for h in HORIZONS:
            by_horizon[f"H{h}"]={
                "overall":describe_labels(subset,h),
                "by_symbol":{
                    symbol:describe_labels(subset.loc[subset.symbol.eq(symbol)],h)
                    for symbol in SEEN_SYMBOLS
                },
            }
        rows.append({
            "fold_id":fold["fold_id"],
            "check_start_utc":fold["check_start"].isoformat(),
            "check_end_exclusive_utc":fold["check_end"].isoformat(),
            "rows":len(subset),
            "horizons":by_horizon,
        })
        print(f"fold={fold['fold_id']} label rows={len(subset)}",flush=True)
    summary={}
    for h in HORIZONS:
        hk=f"H{h}"
        counts=np.array([
            [f["horizons"][hk]["overall"]["counts"][name] for name in LABELS]
            for f in rows
        ],dtype=int)
        if np.any(counts.sum(axis=1)==0):
            raise AssertionError("Empty check")
        summary[hk]={
            "folds":len(rows),
            "mean_fold_class_fractions":{
                name:float(np.mean(counts[:,j]/counts.sum(axis=1)))
                for j,name in enumerate(LABELS)
            },
            "pooled_class_fractions_across_overlapping_checks":{
                name:float(counts[:,j].sum()/counts.sum())
                for j,name in enumerate(LABELS)
            },
            "mean_fold_near_barrier_fraction":float(np.mean([
                row["horizons"][hk]["overall"][
                    "near_original_barrier_within_10pct_fraction"] for row in rows
            ])),
            "mean_fold_ret_bps_median":float(np.mean([
                row["horizons"][hk]["overall"]["absolute_future_return_bps"]["median"]
                for row in rows
            ])),
            "threshold_sensitivity_mean_fold_changed_fraction":{
                str(mult):float(np.mean([
                    row["horizons"][hk]["overall"]["thresholds"][str(mult)][
                        "changed_vs_0_5_fraction"] for row in rows
                ])) for mult in MULTIPLIERS
            },
            "by_symbol_mean_fold_movement_fraction":{
                symbol:float(np.mean([
                    row["horizons"][hk]["by_symbol"][symbol]["movement_fraction"]
                    for row in rows
                ])) for symbol in SEEN_SYMBOLS
            },
        }
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_INSPECTED_DEV_CHECKS_NOT_INDEPENDENT",
        "source_dataset_fingerprint_sha256":fingerprint,
        "folds_analyzed":len(rows),
        "last_allowed_check_end_exclusive_utc":cutoff.isoformat(),
        "horizons":list(HORIZONS),
        "label_schema":"vol_scaled_v1",
        "baseline_multiplier":BASELINE_MULTIPLIER,
        "diagnostic_multipliers":list(MULTIPLIERS),
        "no_model_fits":True,
        "ADA_loaded_or_scored":False,
        "fold_OOS_evaluated":False,
        "final90_labels_constructed":False,
        "frozen_models_modified":False,
        "limitations":[
            "Only previously inspected development validation-check observations.",
            "Fold training windows overlap; later development periods may overlap chronological prior folds.",
            "Multiple multipliers relabel same retrospective outcomes; changes do NOT improve model detection.",
            "No oracle-conditioned direction baseline is tradable or known ahead of time.",
            "No feature choice, trained classifier, trading PnL or OOS validation.",
        ],
        "summary":summary,
        "folds":rows,
    }


def main():
    parser=argparse.ArgumentParser(
        description="Read-only H2/H6 label prevalence and fixed-threshold stability audit")
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--fold-plan",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--months",type=int,default=24)
    parser.add_argument("--max-folds",type=int)
    parser.add_argument("--verify-hashes",action="store_true")
    args=parser.parse_args()
    if args.months!=24:
        parser.error("Exactly 24 original monthly datasets required")
    if args.output.exists():
        parser.error("Refusing to overwrite any prior report")
    if args.output.resolve()==args.data_root.resolve() or args.data_root.resolve() in args.output.resolve().parents:
        parser.error("Cannot write under raw P0 data root")
    source=json.loads(args.fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(source)  # reject invalid plan BEFORE loading any private candles
    _=plan
    store=CandleStore(args.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Reading immutable seen asset",symbol,flush=True)
        candles[symbol],parts=load_symbol(
            store,symbol,months=args.months,verify_hashes=args.verify_hashes)
        manifests.extend(parts)
    report=run_audit(
        candles,source,fingerprint=manifest_fingerprint(manifests),
        max_folds=args.max_folds)
    report["fold_plan_sha256"]=hashlib.sha256(args.fold_plan.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved descriptive label stability report:",args.output.resolve(),flush=True)
    for key,summary in report["summary"].items():
        cls=summary["mean_fold_class_fractions"]
        print(
            f"{key} mean_fold DOWN={100*cls['DOWN']:.2f}% "
            f"NEUTRAL={100*cls['NEUTRAL']:.2f}% UP={100*cls['UP']:.2f}% "
            f"near_boundary={100*summary['mean_fold_near_barrier_fraction']:.2f}%",
            flush=True,
        )


if __name__=="__main__":
    main()
