"""One locked H6 30-vs-25 out-of-time comparison; never trains models.

Read separately frozen model metadata, pre-maturity forecasts and newly
matured H6 outcome labels. Refuse any historical row, duplicate, missing
paired observation or future label. Historical/OOS/ADA data is never loaded.

This is an OFFLINE SCORER ONLY. It cannot prove that prediction timestamps
or model hashes are authentic; immutable external artifacts are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.confidence_diagnostics import decision_and_margin, directional_metrics
from mlpipeline.evaluation import evaluate
from mlpipeline.feature_group_ablation import GROUPS, validate_feature_groups
from mlpipeline.features import FEATURE_COLUMNS

PROTOCOL = "h6_30_vs_25_forward_preregistered_v1"
START = pd.Timestamp("2026-10-17T00:00:00Z")
END = pd.Timestamp("2026-11-14T00:00:00Z")
DAYS = 28
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
VARIANTS = ("baseline_30", "no_candle_shape_25")
REQUIRED_PRED = ("symbol", "bar_close_at", "predicted_at_utc", "variant",
                 "p_down", "p_neutral", "p_up")
REQUIRED_LABEL = ("symbol", "bar_close_at", "label_mature_at", "target_h6")
EXPECTED_PER_SYMBOL = DAYS * 24 * 12
MIN_COMPLETENESS = 0.95
MIN_UPLIFT_GAIN_POINTS = 0.50
MIN_COVERAGE_FRACTION = 0.30
MAX_COVERAGE_FRACTION = 0.50
MIN_COVERAGE_DELTA_POINTS = -2.0


def columns_for_models() -> dict[str, tuple[str, ...]]:
    validate_feature_groups()
    removed = set(GROUPS["candle_shape"])
    full = tuple(FEATURE_COLUMNS)
    candidate = tuple(x for x in full if x not in removed)
    if len(full) != 30 or len(candidate) != 25 or len(removed) != 5:
        raise ValueError("Historical candle_shape schema changed")
    return dict(zip(VARIANTS, (full, candidate)))


def _utc(value, name):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or str(timestamp.tz_convert("UTC").tz) != "UTC":
        raise ValueError(f"{name} must contain timezone-aware timestamps")
    return timestamp.tz_convert("UTC")


def validate_manifest(manifest: dict) -> dict:
    """Metadata must describe genuinely frozen upstream models/thresholds."""
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Wrong locked prospective protocol")
    if manifest.get("horizon") != 6 or manifest.get("coverage_target") != 0.40:
        raise ValueError("Wrong horizon or frozen threshold policy")
    if (manifest.get("uses_ada") is not False
            or manifest.get("uses_fold_oos") is not False
            or manifest.get("uses_reserved_final90") is not False):
        raise ValueError("Sealed data cannot be used for freezing")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(VARIANTS):
        raise ValueError("Exactly two frozen model manifests required")
    expected = columns_for_models()
    for variant, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError("Invalid model metadata")
        if tuple(spec.get("feature_columns", ())) != expected[variant]:
            raise ValueError(f"Frozen column mismatch: {variant}")
        digest = spec.get("model_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError("SHA256 of serialized frozen model required")
        threshold = spec.get("threshold")
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("Invalid frozen abstention threshold")
        counts = spec.get("train_class_counts")
        if (not isinstance(counts, list) or len(counts) != 3
                or any(type(v) is not int or v <= 0 for v in counts)):
            raise ValueError("Valid frozen training priors required")
        if spec.get("tuning_feasible") is not True:
            raise ValueError("Frozen threshold was not tuning-feasible")
        for field in ("model_frozen_at_utc", "last_training_label_mature_at_utc",
                      "last_tuning_bar_close_at_utc"):
            if _utc(spec.get(field), field) >= START:
                raise ValueError("Freeze and tuning must precede prospective start")
    return models


def _parse_utc_column(frame:pd.DataFrame, name:str):
    stamps = pd.to_datetime(frame[name], utc=True, errors="raise")
    if stamps.isna().any():
        raise ValueError(f"Null timestamp in {name}")
    # A CSV with naive dates is not acceptable, even if pandas can coerce it.
    if not frame[name].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$",regex=True).all():
        raise ValueError(f"Explicit UTC offset required in {name}")
    return stamps


def validate_rows(forecasts:pd.DataFrame, labels:pd.DataFrame, *,as_of_utc):
    if not set(REQUIRED_PRED).issubset(forecasts.columns):
        raise ValueError("Missing forecast fields")
    if not set(REQUIRED_LABEL).issubset(labels.columns):
        raise ValueError("Missing label fields")
    if forecasts.empty or labels.empty:
        raise ValueError("Future data not yet available")
    if set(labels.symbol) != set(SYMBOLS) or set(forecasts.symbol) != set(SYMBOLS):
        raise ValueError("Only BTC ETH SOL XRP, all four required")
    if set(forecasts.variant) != set(VARIANTS):
        raise ValueError("Both predeclared variants required")
    if forecasts.duplicated(["symbol","bar_close_at","variant"]).any():
        raise ValueError("Duplicate forecast keys")
    if labels.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate future outcome keys")
    f=forecasts.copy()
    l=labels.copy()
    f["bar_close_at"]=_parse_utc_column(f,"bar_close_at")
    l["bar_close_at"]=_parse_utc_column(l,"bar_close_at")
    f["predicted_at_utc"]=_parse_utc_column(f,"predicted_at_utc")
    l["label_mature_at"]=_parse_utc_column(l,"label_mature_at")
    now=_utc(as_of_utc,"as_of_utc")
    if now < END + pd.Timedelta(minutes=30):
        raise ValueError("Full 28-day outcome window is not yet mature")
    if now < l.label_mature_at.max():
        raise ValueError("Future labels not mature as of analysis")
    if ((l.bar_close_at < START) | (l.bar_close_at >= END)).any():
        raise ValueError("Outcomes outside locked prospective window")
    if ((f.bar_close_at < START) | (f.bar_close_at >= END)).any():
        raise ValueError("Forecasts outside locked prospective window")
    if not (l.label_mature_at == l.bar_close_at + pd.Timedelta(minutes=30)).all():
        raise ValueError("H6 requires 30-minute causal maturity")
    if ((f.predicted_at_utc < f.bar_close_at)
            | (f.predicted_at_utc >= f.bar_close_at + pd.Timedelta(minutes=30))).any():
        raise ValueError("Predictions must have been recorded before H6 maturity")
    if not l.target_h6.isin((0,1,2)).all():
        raise ValueError("H6 labels outside 0,1,2")
    for symbol in SYMBOLS:
        rows=l.loc[l.symbol.eq(symbol)]
        if len(rows) < int(np.ceil(EXPECTED_PER_SYMBOL * MIN_COMPLETENESS)):
            raise ValueError(f"Insufficient complete prospective data: {symbol}")
        if (not set(pd.date_range(START,END-pd.Timedelta(days=1),
                                    freq="D")).issubset(set(
                rows.bar_close_at.dt.floor("D")))):
            raise ValueError(f"Missing calendar day for {symbol}")
    cols=list(("p_down","p_neutral","p_up"))
    p=f[cols].to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1),1,atol=1e-5)):
        raise ValueError("Forecast probabilities must be 3-class distributions")
    left=set(zip(l.symbol,l.bar_close_at))
    for variant in VARIANTS:
        right=set(zip(f.loc[f.variant.eq(variant),"symbol"],
                      f.loc[f.variant.eq(variant),"bar_close_at"]))
        if left!=right:
            raise ValueError(f"Unequal future rows for variant {variant}")
    return f,l


def _score(frame:pd.DataFrame,spec:dict):
    probs=frame[["p_down","p_neutral","p_up"]].to_numpy(dtype=float)
    pred,margin=decision_and_margin(
        probs,np.array(spec["train_class_counts"]),alpha=1.0)
    metrics=directional_metrics(
        frame.target_h6.to_numpy(dtype=int),pred,margin,
        threshold=float(spec["threshold"]))
    metrics["raw_log_loss"]=float(evaluate(
        frame.target_h6.to_numpy(dtype=int),probs)["log_loss"])
    return metrics


def compare(forecasts:pd.DataFrame,labels:pd.DataFrame,manifest:dict,*,as_of_utc):
    models=validate_manifest(manifest)
    f,l=validate_rows(forecasts,labels,as_of_utc=as_of_utc)
    scored={}
    by_symbol={}
    daily={}
    for variant in VARIANTS:
        joined=f.loc[f.variant.eq(variant)].merge(
            l,on=["symbol","bar_close_at"],validate="one_to_one")
        joined=joined.sort_values(["symbol","bar_close_at"])
        scored[variant]=_score(joined,models[variant])
        by_symbol[variant]={
            symbol:_score(joined.loc[joined.symbol.eq(symbol)],models[variant])
            for symbol in SYMBOLS
        }
        daily[variant]={
            day:_score(joined.loc[joined.bar_close_at.dt.floor("D").eq(day)],
                       models[variant])
            for day in pd.date_range(START,END-pd.Timedelta(days=1),freq="D")
        }
    base,candidate=(scored[v] for v in VARIANTS)
    dates=list(daily[VARIANTS[0]])
    diffs=np.array([
        daily[VARIANTS[1]][d]["uplift_points"]
        - daily[VARIANTS[0]][d]["uplift_points"] for d in dates
    ],dtype=float)
    if not np.isfinite(diffs).all():
        raise ValueError("Insufficient daily directional decisions")
    rng=np.random.default_rng(42)
    bootstrap=diffs[rng.integers(0,len(diffs),size=(2000,len(diffs)))].mean(axis=1)
    ci=[float(v) for v in np.quantile(bootstrap,[0.025,0.975])]
    uplift_delta=candidate["uplift_points"]-base["uplift_points"]
    coverage_delta=100*(candidate["coverage"]-base["coverage"])
    log_loss_improvement=base["raw_log_loss"]-candidate["raw_log_loss"]
    by_symbol_delta={
        symbol:by_symbol[VARIANTS[1]][symbol]["uplift_points"]
              - by_symbol[VARIANTS[0]][symbol]["uplift_points"]
        for symbol in SYMBOLS
    }
    gates={
        "pooled_uplift_gain_at_least_0_5_points":uplift_delta>=MIN_UPLIFT_GAIN_POINTS,
        "mean_daily_delta_95pct_day_bootstrap_lower_above_zero":ci[0]>0,
        "candidate_positive_uplift_gain_on_all_four_assets":
            all(v is not None and v>0 for v in by_symbol_delta.values()),
        "pooled_log_loss_not_worse":log_loss_improvement>=0,
        "candidate_coverage_drop_no_more_than_2_points":
            coverage_delta>=MIN_COVERAGE_DELTA_POINTS,
        "both_coverages_between_30_and_50pct":
            all(MIN_COVERAGE_FRACTION<=m["coverage"]<=MAX_COVERAGE_FRACTION
                for m in (base,candidate)),
    }
    return {
        "protocol":PROTOCOL,
        "window_start_utc":START.isoformat(),
        "window_end_exclusive_utc":END.isoformat(),
        "rows_per_symbol":l.groupby("symbol").size().to_dict(),
        "primary_pooled_uplift_gain_points":uplift_delta,
        "mean_daily_uplift_gain_points":float(diffs.mean()),
        "daily_bootstrap_95pct_ci_points":ci,
        "pooled_realized_coverage_delta_points":coverage_delta,
        "pooled_log_loss_improvement":log_loss_improvement,
        "per_symbol_uplift_delta_points":by_symbol_delta,
        "pooled_metrics":scored,
        "gates":gates,
        "all_predeclared_gates_pass":all(gates.values()),
        "decision":"MANUAL_REVIEW_ONLY_NEVER_AUTOMATIC_PROMOTION",
        "evidence_limitations":[
            "Artifact metadata and forecast timestamps are self-reported CSV/JSON.",
            "Timestamp truth requires an externally immutable prediction log.",
            "Scores derived from post-cutoff data are independent only if model and threshold were frozen first.",
            "An out-of-time offline replay is not necessarily prospectively logged forecasting.",
            "No claims about market returns, execution, or profitability.",
        ],
    }


def _sha256_file(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description="Score locked future H6 30/25 comparison")
    p.add_argument("--models-manifest",required=True,type=Path)
    p.add_argument("--forecasts-csv",required=True,type=Path)
    p.add_argument("--labels-csv",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite prior prospective evaluation")
    with args.models_manifest.open(encoding="utf-8") as f:
        manifest=json.load(f)
    forecasts=pd.read_csv(args.forecasts_csv)
    labels=pd.read_csv(args.labels_csv)
    result=compare(
        forecasts,labels,manifest,
        as_of_utc=pd.Timestamp.now(tz="UTC"),
    )
    result["input_sha256"]={
        "models_manifest":_sha256_file(args.models_manifest),
        "forecasts_csv":_sha256_file(args.forecasts_csv),
        "labels_csv":_sha256_file(args.labels_csv),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved locked forward H6 result",args.output.resolve(),flush=True)


if __name__=="__main__":
    main()
