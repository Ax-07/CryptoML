"""Unlabeled frequency census of causal EMA9/26 pullbacks on pre-July P0 data.

Research only: never loads ADA, July+ 2026, P2 reports, OOS scores or P3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-12-01T00:00:00Z")
CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
PROTOCOL = "trend_pullback_v1_prejuly_unlabeled_frequency"
METRICS = (
    "closed_bars", "valid_indicators", "uptrend_prior_bars",
    "first_ema9_touch_bars", "raw_candidates", "cooldown_selected",
)


def count_features(features: pd.DataFrame) -> dict:
    """Count current/past-only signals; do not read returns or future outcomes."""
    needed = {
        "symbol", "bar_close_at", "ema9", "ema26", "atr14",
        "uptrend_prior", "first_ema9_touch", "pullback_raw", "pullback_signal",
    }
    if needed.difference(features.columns):
        raise ValueError("Missing causal detector columns")
    if features.empty or features.symbol.nunique() != 1:
        raise ValueError("Expected one nonempty symbol")
    symbol = str(features.symbol.iloc[0])
    if symbol not in SYMBOLS:
        raise ValueError("Unapproved symbol in research census")
    times = features.bar_close_at
    if (not isinstance(times.dtype, pd.DatetimeTZDtype)
            or str(times.dt.tz) != "UTC" or times.isna().any()
            or not times.is_unique or not times.is_monotonic_increasing
            or not times.ge(START).all() or not times.lt(CUTOFF).all()):
        raise ValueError("Nonchronological or out-of-scope UTC timestamps")
    indicator = features[["ema9", "ema26", "atr14"]].to_numpy(dtype=float)
    if np.isinf(indicator).any():
        raise ValueError("Nonfinite indicator values")
    for flag in ("uptrend_prior", "first_ema9_touch", "pullback_raw", "pullback_signal"):
        if not pd.api.types.is_bool_dtype(features[flag]) or features[flag].isna().any():
            raise ValueError(f"Invalid boolean flag {flag}")
    raw = features.pullback_raw.to_numpy(dtype=bool)
    chosen = features.pullback_signal.to_numpy(dtype=bool)
    if np.any(chosen & ~raw):
        raise ValueError("Cooldown selection outside raw events")
    valid = np.isfinite(indicator).all(axis=1) & (indicator[:, 2] > 0)
    if np.any(raw & ~valid):
        raise ValueError("Raw signal with incomplete indicators")

    def _counts(frame: pd.DataFrame) -> dict:
        values = frame[["ema9", "ema26", "atr14"]].to_numpy(dtype=float)
        valid_count = int((np.isfinite(values).all(axis=1) & (values[:, 2] > 0)).sum())
        n = int(len(frame))
        selected = int(frame.pullback_signal.sum())
        raw_count = int(frame.pullback_raw.sum())
        return {
            "closed_bars": n,
            "valid_indicators": valid_count,
            "uptrend_prior_bars": int(frame.uptrend_prior.sum()),
            "first_ema9_touch_bars": int(frame.first_ema9_touch.sum()),
            "raw_candidates": raw_count,
            "cooldown_selected": selected,
            "suppressed_by_cooldown": raw_count - selected,
            "selected_per_1000_bars": 1000 * selected / n if n else None,
            "selected_per_1000_valid": 1000 * selected / valid_count if valid_count else None,
        }

    months = []
    for month, part in features.groupby(times.dt.strftime("%Y-%m"), sort=True):
        months.append({"month": str(month), **_counts(part)})
    overall = _counts(features)
    for metric in METRICS:
        if sum(row[metric] for row in months) != overall[metric]:
            raise AssertionError(f"Monthly census does not reconcile: {metric}")
    return {"symbol": symbol,
            "first_bar_close_utc": times.iloc[0].isoformat(),
            "last_bar_close_utc": times.iloc[-1].isoformat(),
            "all": overall, "by_month": months}


def frequency_report(candles: dict, fingerprint: str) -> dict:
    """Generate four independent per-symbol summaries from causal detector output."""
    if set(candles) != set(SYMBOLS) or len(fingerprint) != 64:
        raise ValueError("Wrong approved four-asset P0 inputs/fingerprint")
    from mlpipeline.trend_pullback import detect_pullbacks
    results = {}
    for symbol in SYMBOLS:
        frame = candles[symbol]
        if (frame.empty or frame.symbol.nunique() != 1 or frame.symbol.iloc[0] != symbol
                or not frame.bar_close_at.lt(CUTOFF).all()
                or not frame.bar_close_at.ge(START).all()):
            raise ValueError("Candle input outside approved historical range")
        results[symbol] = count_features(detect_pullbacks(frame))
    return {
        "protocol": PROTOCOL,
        "status": "EXPLORATORY_UNLABELED_FREQUENCY_ONLY_NOT_OOS",
        "source_dataset_fingerprint_sha256": fingerprint,
        "scope_start_inclusive_utc": START.isoformat(),
        "scope_end_exclusive_utc": CUTOFF.isoformat(),
        "assets": list(SYMBOLS),
        "read_only": True,
        "does_not_load_reserved_data_or_labels": True,
        "no_training_no_return_scoring_no_trading": True,
        "note": "Counts are descriptive on reused pre-July historical research data, not an independent test.",
        "by_asset": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 EMA pullback frequency census (UNLABELED)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    a = parser.parse_args()
    root = a.data_root.resolve()
    outs = [a.output_json.resolve(), a.output_csv.resolve()]
    if (outs[0] == outs[1]
            or any(p.exists() or not p.parent.is_dir()
                   or p.parent != outs[0].parent
                   or p.is_relative_to(root.parent) for p in outs)):
        parser.error("Use two NEW files in one existing isolated results directory outside P0")
    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint
    from mlpipeline.prospective_model_pack import load_clean_historic

    # Approved loader verifies SHA256 and reads ONLY Dec 2025--Jun 2026 monthly Parquets.
    candles, manifests = load_clean_historic(CandleStore(root), verify_hashes=True)
    report = frequency_report(candles, manifest_fingerprint(manifests))
    rows = [{"symbol": symbol, "month": item["month"],
             **{key: item[key] for key in (*METRICS, "suppressed_by_cooldown",
                                             "selected_per_1000_bars", "selected_per_1000_valid")}}
            for symbol, result in report["by_asset"].items()
            for item in result["by_month"]]
    # JSON is the completion marker; never overwrite an earlier research result.
    with a.output_csv.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)
    with a.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for symbol, result in report["by_asset"].items():
        overall = result["all"]
        print(f"{symbol}: {overall['cooldown_selected']} confirmed candidates "
              f"of {overall['closed_bars']} closed bars "
              f"({overall['selected_per_1000_bars']:.3f} per 1000 bars)")
    print("UNLABELED RESEARCH ONLY - no returns, no model fitting, no OOS")
    print("Saved:", a.output_json, a.output_csv)


if __name__ == "__main__":
    main()
