"""P4 V2 joint trend-strength × pullback-depth census, WITHOUT returns.

Cross-tabulates the preregistered V2 ATR buckets on V1-selected,
closed-M5 signals. Reads only approved Dec 2025--Jun 2026 P0 data.
No fitting, candidate selection, performance measurement, or held-out read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.pullback_frequency import SYMBOLS
from mlpipeline.pullback_regime_v2 import (
    FEATURE_COLUMNS, describe_regimes,
)
from mlpipeline.pullback_regime_census_v2 import (
    DEPTH, MONTHS, STRENGTH, summarize_regime_features,
)

PROTOCOL = "p4_v2_strength_depth_joint_unlabeled_v1"


def cross_counts(features: pd.DataFrame) -> dict:
    """Explicit-zero 3×4 count matrix; no labels or outcomes allowed."""
    needed = {"pullback_signal", *FEATURE_COLUMNS}
    if not needed.issubset(features.columns):
        raise ValueError("Missing causal V2 feature fields")
    if not pd.api.types.is_bool_dtype(features.pullback_signal):
        raise ValueError("Signal flag must be boolean")
    forbidden = {"gross_bps", "net_2bps", "net_5bps", "net_10bps",
                 "label_mature_at", "entry_price", "exit_price"}
    if forbidden.intersection(features.columns):
        raise ValueError("Outcome fields forbidden in unlabeled census")

    chosen = features.loc[features.pullback_signal]
    matrix = chosen.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    eligible = chosen.loc[valid]
    strength_bins = pd.cut(
        eligible.v2_trend_spread_atr,
        [-np.inf, 0.5, 1.5, np.inf],
        labels=list(STRENGTH), right=False,
    )
    depth_bins = pd.cut(
        eligible.v2_pullback_depth_ema9_atr,
        [-np.inf, 0.0, 0.5, 1.0, np.inf],
        labels=list(DEPTH), right=False,
    )
    table = pd.crosstab(strength_bins, depth_bins, dropna=False)
    table = table.reindex(index=list(STRENGTH), columns=list(DEPTH),
                          fill_value=0)
    cells = {
        s: {d: int(table.loc[s, d]) for d in DEPTH}
        for s in STRENGTH
    }
    events, valid_events = len(chosen), len(eligible)
    if (sum(sum(v.values()) for v in cells.values()) != valid_events
            or any(cells[s]["no_undercut"] for s in STRENGTH)):
        raise AssertionError("Non-reconciling or V1-impossible signal bucket")
    return {
        "events": int(events), "valid_events": int(valid_events),
        "unclassified": int(events-valid_events), "joint": cells,
    }


def joint_report(features_by_symbol: dict[str, pd.DataFrame],
                 fingerprint: str) -> tuple[dict, pd.DataFrame]:
    """Check against existing independently computed descriptive marginals."""
    marginal_report, _ = summarize_regime_features(features_by_symbol,
                                                    fingerprint)
    results = {}
    rows = []
    for symbol in SYMBOLS:
        features = features_by_symbol[symbol]
        monthly = {}
        for month, sub in features.groupby(
            features.bar_close_at.dt.strftime("%Y-%m"), sort=True
        ):
            joint = cross_counts(sub)
            marginals = marginal_report["by_asset"][symbol]["by_month"][month]
            _compare(joint, marginals)
            monthly[month] = joint
            for strength in STRENGTH:
                for depth in DEPTH:
                    rows.append({
                        "symbol": symbol, "month": month,
                        "strength": strength, "depth": depth,
                        "events": joint["joint"][strength][depth],
                    })
        if tuple(monthly) != MONTHS:
            raise AssertionError(f"Missing monthly P0 partitions: {symbol}")
        total = cross_counts(features)
        _compare(total, marginal_report["by_asset"][symbol]["total"])
        for s in STRENGTH:
            for d in DEPTH:
                if sum(monthly[m]["joint"][s][d] for m in MONTHS) != total["joint"][s][d]:
                    raise AssertionError(f"Monthly count mismatch: {symbol}/{s}/{d}")
        results[symbol] = {"total": total, "by_month": monthly}
    grand_joint = {
        s: {d: sum(results[x]["total"]["joint"][s][d] for x in SYMBOLS)
            for d in DEPTH}
        for s in STRENGTH
    }
    grand = {
        "events": sum(results[x]["total"]["events"] for x in SYMBOLS),
        "valid_events": sum(results[x]["total"]["valid_events"] for x in SYMBOLS),
        "unclassified": sum(results[x]["total"]["unclassified"] for x in SYMBOLS),
        "joint": grand_joint,
    }
    _compare(grand, marginal_report["total"])
    report = {
        "protocol": PROTOCOL,
        "status": "EXPLORATORY_UNLABELED_JOINT_COUNTS_NOT_OOS",
        "source_dataset_fingerprint_sha256": fingerprint,
        "symbols": list(SYMBOLS),
        "strength_bins_atr": {"low": "<0.5", "mid": "[0.5,1.5)",
                              "high": ">=1.5"},
        "depth_bins_atr": {"no_undercut": "<0", "shallow": "[0,0.5)",
                           "medium": "[0.5,1)", "deep": ">=1"},
        "note": "No future labels, outcomes, strategy ranking or independent validation.",
        "total": grand, "by_asset": results,
    }
    return report, pd.DataFrame(rows)


def _compare(joint: dict, marginal: dict) -> None:
    for field in ("events", "valid_events", "unclassified"):
        if joint[field] != marginal[field]:
            raise AssertionError(f"Mismatch between joint and marginal {field}")
    for s in STRENGTH:
        if sum(joint["joint"][s].values()) != marginal["strength"][s]:
            raise AssertionError(f"Strength marginal mismatch: {s}")
    for d in DEPTH:
        if sum(joint["joint"][s][d] for s in STRENGTH) != marginal["depth"][d]:
            raise AssertionError(f"Depth marginal mismatch: {d}")


def main() -> None:
    p = argparse.ArgumentParser(description="Unlabeled P4 V2 cross tab ONLY")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()
    root = args.data_root.resolve()
    out_json, out_csv = args.output_json.resolve(), args.output_csv.resolve()
    if (out_json == out_csv or out_json.parent != out_csv.parent
            or not out_json.parent.is_dir()
            or any(path.exists() or path.is_relative_to(root)
                   for path in (out_json, out_csv))):
        p.error("Require two NEW distinct output files in an existing folder outside P0")

    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint
    from mlpipeline.prospective_model_pack import load_clean_historic
    candles, manifests = load_clean_historic(CandleStore(root),
                                            verify_hashes=True)
    if set(candles) != set(SYMBOLS):
        raise ValueError("Only the four seen development symbols are permitted")
    features = {s: describe_regimes(candles[s]) for s in SYMBOLS}
    report, table = joint_report(features, manifest_fingerprint(manifests))
    with out_csv.open("x", newline="", encoding="utf-8") as f:
        table.to_csv(f, index=False)
    with out_json.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"All events: {report['total']['events']} "
          f"(unclassified: {report['total']['unclassified']})")
    for s in STRENGTH:
        print(s, report["total"]["joint"][s])
    print("UNLABELED COUNTS ONLY — no outcomes, model training or OOS")
    print("Saved", out_json, out_csv)


if __name__ == "__main__":
    main()
