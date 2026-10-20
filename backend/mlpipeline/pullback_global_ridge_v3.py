"""P4 V3 21-month single-shot global Ridge comparison (research only).

Four seen symbols, SHA-verified Oct 2024--Jun 2026 P0 candles; do NOT touch
ADA, Jul--Sep 2026 or any P3 scores. Causal features at t/t+1, common
hypothetical t+2 entry. Compare 9-feature short/long and 19-feature long
Ridge against B, matched-coverage random expectation, and zero trades.
All diagnostics retrospective and already-explored; never independent OOS.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlpipeline.pullback_long_history_v3 import (
    CUTOFF, SYMBOLS, load_long_development_history,
)
from mlpipeline.pullback_outcomes import research_outcomes
from mlpipeline.pullback_baselines import _score
from mlpipeline.pullback_ml import (
    FEATURES as V1_FEATURES, confirmed_features, LABEL_CLIP_BPS,
    RIDGE_ALPHA, APRIL_KEEP_QUANTILE,
)
from mlpipeline.pullback_regime_v2 import (
    FEATURE_COLUMNS as REGIME_FEATURES, describe_regimes,
)
from mlpipeline.trend_pullback import M5, _validate_closed_m5

PROTOCOL = "p4_v3_21month_fixed_ridge_global_9v19_v1"
START = pd.Timestamp("2024-10-01T00:00:00Z")
SHORT_START = pd.Timestamp("2025-12-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2026-04-01T00:00:00Z")
APRIL_END = pd.Timestamp("2026-05-01T00:00:00Z")
MAY_END = pd.Timestamp("2026-06-01T00:00:00Z")
MATURITY = pd.Timedelta(minutes=35)
MODEL_SPECS = {
    "C0_ridge9_short4m": (SHORT_START, V1_FEATURES),
    "C1_ridge9_long18m": (START, V1_FEATURES),
    "C2_ridge19_long18m": (START, V1_FEATURES + REGIME_FEATURES),
}
STRATEGIES = ("A_all", "B_confirmed", *MODEL_SPECS, "D_no_trade")
COSTS = (2, 5, 10)


def annotate_confirmation_long(
    candles: pd.DataFrame, detected: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Exact V1 B rule, without V1's December-only START restriction.

    B is fully observed at close(t+1), hypothetical entry at open(t+2).
    No outcome value is involved in deciding B.
    """
    required = {"signal_close_at", "entry_at", "mature_at"}
    if (not required.issubset(outcomes.columns)
            or len(candles) != len(detected)
            or not candles.bar_close_at.equals(detected.bar_close_at)
            or not candles.bar_close_at.is_unique
            or not outcomes.signal_close_at.is_unique):
        raise ValueError("Misaligned or duplicated V3 data")
    if not outcomes.empty and (
        not outcomes.signal_close_at.ge(START).all()
        or not outcomes.mature_at.lt(CUTOFF).all()
    ):
        raise ValueError("V3 outcomes outside approved development")
    positions = {stamp: i for i, stamp in enumerate(detected.bar_close_at)}
    confirmed = []
    for row in outcomes.itertuples(index=False):
        i = positions.get(row.signal_close_at)
        if i is None or not bool(detected.pullback_signal.iloc[i]):
            raise ValueError("Outcome has no V1 pullback signal")
        if (i+2 >= len(candles)
                or detected.bar_close_at.iloc[i+1] != row.signal_close_at + M5
                or row.entry_at != row.signal_close_at + M5
                or candles.bar_open_at.iloc[i+2] != row.entry_at
                or candles.bar_close_at.iloc[i+1] != row.entry_at
                or row.mature_at != row.signal_close_at + MATURITY):
            raise ValueError("Incorrect B timing or maturity")
        bar = candles.iloc[i+1]
        ema9 = float(detected.ema9.iloc[i+1])
        confirmed.append(bool(
            np.isfinite(ema9)
            and bar.close > bar.open
            and bar.close > ema9
            and bar.close > candles.close.iloc[i]
        ))
    events = outcomes.copy().reset_index(drop=True)
    events["baseline_b_confirmed"] = np.asarray(confirmed, dtype=bool)
    return events


def prepare_symbol(candles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construct labels separately, then join ONLY t/t+1 input features.

    V1 nine features are reused unchanged; V2 ten are frozen at signal t.
    All later outcome fields stay outside the fitted feature matrices.
    """
    bars = _validate_closed_m5(candles)
    if (bars.symbol.iloc[0] not in SYMBOLS
            or not bars.bar_close_at.ge(START).all()
            or not bars.bar_close_at.lt(CUTOFF).all()):
        raise ValueError("Input outside approved 21-month four-symbol scope")
    causal = describe_regimes(bars)
    total_signals = int(causal.pullback_signal.sum())
    outcomes = research_outcomes(bars, development_end=CUTOFF)
    # The theoretical labels are hindsight-only, not part of any input feature.
    events = annotate_confirmation_long(bars, causal, outcomes)
    events = confirmed_features(bars, causal, events)
    detected_signals = causal.loc[
        causal.pullback_signal, ["symbol", "bar_close_at", *REGIME_FEATURES]
    ].copy()
    detected_signals = detected_signals.rename(
        columns={"bar_close_at": "signal_close_at"}
    )
    if (detected_signals.signal_close_at.duplicated().any()
            or len(detected_signals) != total_signals):
        raise AssertionError("Duplicate V2 signal features")
    events = events.merge(
        detected_signals, on=["symbol", "signal_close_at"],
        validate="one_to_one", how="left", sort=False,
    )
    if len(events) != len(outcomes):
        raise AssertionError("Missing/duplicated feature join")
    if not events.empty and (not events.signal_close_at.is_unique
                            or not events.signal_close_at.is_monotonic_increasing):
        raise AssertionError("Unsorted or duplicated V3 outcomes")

    selected = events.loc[events.baseline_b_confirmed]
    all_features = list(V1_FEATURES + REGIME_FEATURES)
    if not selected.empty and not np.isfinite(
        selected.loc[:, all_features].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing B feature in common 9/19-feature cohort")
    return events, {
        "symbol": str(bars.symbol.iloc[0]),
        "closed_bars": int(len(bars)),
        "signals_unlabeled": total_signals,
        "matured_A": int(len(events)),
        "confirmed_B": int(events.baseline_b_confirmed.sum()),
    }


def _partition(frame: pd.DataFrame, start: pd.Timestamp,
               end: pd.Timestamp) -> pd.DataFrame:
    if not START <= start < end <= CUTOFF:
        raise ValueError("Partition outside approved development history")
    # Excludes events whose full 35-minute horizon crosses a partition.
    return frame.loc[
        frame.signal_close_at.ge(start)
        & frame.signal_close_at.lt(end)
        & frame.mature_at.lt(end)
    ].copy()


def _validate_and_combine(events_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if set(events_by_symbol) != set(SYMBOLS):
        raise ValueError("Four seen symbols only, excluding ADA")
    chunks = []
    expected = {
        "symbol", "signal_close_at", "entry_at", "mature_at",
        "baseline_b_confirmed", "net_2bps", "net_5bps", "net_10bps",
        "exit_reason", *V1_FEATURES, *REGIME_FEATURES,
    }
    for symbol in SYMBOLS:
        f = events_by_symbol[symbol].copy()
        if (not expected.issubset(f.columns)
                or f.empty or not f.symbol.eq(symbol).all()
                or not f.signal_close_at.is_unique
                or not f.signal_close_at.is_monotonic_increasing
                or not pd.api.types.is_bool_dtype(f.baseline_b_confirmed)):
            raise ValueError(f"{symbol}: invalid event frame")
        for col in ("signal_close_at", "entry_at", "mature_at"):
            t = f[col]
            if (not isinstance(t.dtype, pd.DatetimeTZDtype)
                    or str(t.dt.tz) != "UTC" or t.isna().any()):
                raise ValueError(f"{symbol}: invalid {col} UTC")
        if (not f.signal_close_at.ge(START).all()
                or not f.mature_at.lt(CUTOFF).all()
                or not f.entry_at.eq(f.signal_close_at + M5).all()
                or not f.mature_at.eq(f.signal_close_at + MATURITY).all()):
            raise ValueError(f"{symbol}: out-of-scope or misclocked outcome")
        for cost in COSTS:
            key = f"net_{cost}bps"
            if not np.isfinite(f[key].to_numpy(dtype=float)).all():
                raise ValueError(f"{symbol}: nonfinite net outcome")
        b = f.loc[f.baseline_b_confirmed, V1_FEATURES + REGIME_FEATURES]
        if not np.isfinite(b.to_numpy(dtype=float)).all():
            raise ValueError(f"{symbol}: incomplete common model features")
        chunks.append(f)
    return pd.concat(chunks, ignore_index=True)


def _fit_fixed_ridge(train: pd.DataFrame, features: tuple[str, ...]):
    x = train.loc[:, features].to_numpy(dtype=float)
    y = train.net_5bps.to_numpy(dtype=float)
    if len(train) < 100 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Insufficient/invalid fixed Ridge training events")
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(x, np.clip(y, -LABEL_CLIP_BPS, LABEL_CLIP_BPS))
    return model


def _metrics(group: pd.DataFrame, selected: np.ndarray) -> dict:
    selected = np.asarray(selected, dtype=bool)
    if len(selected) != len(group):
        raise ValueError("Mask does not cover every opportunity")
    return _score(group.iloc[np.flatnonzero(selected)],
                  total_opportunities=len(group))


def _random_expectation(group: pd.DataFrame, selected_count: int) -> dict:
    """Analytic finite-population expectation, same B pool AND trade count.

    Unlike simply comparing net/opp to B, this checks whether selected
    trades are BETTER on average than a random sample from B in that
    exact asset×month group. No Monte Carlo nor p-values.
    """
    b = group.loc[group.baseline_b_confirmed]
    if selected_count > len(b) or selected_count < 0:
        raise ValueError("Selected count cannot exceed confirmed B pool")
    return {
        f"expected_sum_net_{cost}bps": (
            float(selected_count * b[f"net_{cost}bps"].mean())
            if len(b) else 0.0
        )
        for cost in COSTS
    }


def compare_fixed_models(
    events_by_symbol: dict[str, pd.DataFrame], fingerprint: str,
    census_by_symbol: dict[str, dict] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """One prespecified 4m/18m 9/19-feature research comparison."""
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("Invalid provenance fingerprint")
    events = _validate_and_combine(events_by_symbol)
    training = {
        name: _partition(events, start, TRAIN_END).loc[
            lambda x: x.baseline_b_confirmed
        ]
        for name, (start, _) in MODEL_SPECS.items()
    }
    april = _partition(events, TRAIN_END, APRIL_END)
    april_b = april.loc[april.baseline_b_confirmed]
    if len(april_b) < 30:
        raise ValueError("Too few April B examples for score calibration")
    models = {}
    model_details = {}
    for name, (start, features) in MODEL_SPECS.items():
        fit = training[name]
        model = _fit_fixed_ridge(fit, features)
        april_scores = model.predict(
            april_b.loc[:, features].to_numpy(dtype=float)
        )
        threshold = max(
            0.0, float(np.quantile(april_scores, APRIL_KEEP_QUANTILE))
        )
        if not np.isfinite(threshold):
            raise ValueError("Invalid calibration threshold")
        models[name] = (model, features, threshold)
        model_details[name] = {
            "train_from": start.isoformat(),
            "train_to_exclusive": TRAIN_END.isoformat(),
            "train_B_events": int(len(fit)),
            "april_B_score_calibration_events": int(len(april_b)),
            "fitted_target_clip_bps": LABEL_CLIP_BPS,
            "clipped_train_labels": int(
                (fit.net_5bps.abs() > LABEL_CLIP_BPS).sum()
            ),
            "features": list(features),
            "ridge_alpha": RIDGE_ALPHA,
            "threshold_score_net_5bps": threshold,
            "threshold_rule": "prediction > max(0, April score 80th percentile)",
        }

    rows = []
    grouped: dict = {}
    overall_outcomes: dict[str, list[pd.DataFrame]] = {
        key: [] for key in STRATEGIES
    }
    overall_expected: dict[str, dict[int, float]] = {
        name: {cost: 0.0 for cost in COSTS} for name in MODEL_SPECS
    }
    all_n = 0
    for month, start, end in (
        ("2026-05", APRIL_END, MAY_END),
        ("2026-06", MAY_END, CUTOFF),
    ):
        scoped = _partition(events, start, end)
        grouped[month] = {}
        for symbol in SYMBOLS:
            group = scoped.loc[scoped.symbol.eq(symbol)]
            all_n += len(group)
            bmask = group.baseline_b_confirmed.to_numpy(dtype=bool)
            flags: dict[str, np.ndarray] = {
                "A_all": np.ones(len(group), dtype=bool),
                "B_confirmed": bmask,
                "D_no_trade": np.zeros(len(group), dtype=bool),
            }
            for name, (model, columns, threshold) in models.items():
                selection = np.zeros(len(group), dtype=bool)
                if bmask.any():
                    scores = model.predict(
                        group.loc[group.baseline_b_confirmed, columns]
                        .to_numpy(dtype=float)
                    )
                    selection[np.flatnonzero(bmask)] = scores > threshold
                flags[name] = selection

            summary = {}
            for name in STRATEGIES:
                selected = group.iloc[np.flatnonzero(flags[name])]
                stats = _metrics(group, flags[name])
                overall_outcomes[name].append(selected)
                result = dict(stats)
                if name in MODEL_SPECS:
                    baseline = _random_expectation(group, len(selected))
                    for cost in COSTS:
                        expected = baseline[f"expected_sum_net_{cost}bps"]
                        overall_expected[name][cost] += expected
                        result[f"random_expected_net_{cost}bps_per_trade"] = (
                            float(expected/len(selected)) if len(selected) else None
                        )
                        result[f"edge_vs_random_{cost}bps_per_trade"] = (
                            float(stats[f"net_{cost}bps_per_trade"] -
                                  expected/len(selected)) if len(selected) else None
                        )
                        result[f"random_expected_sum_net_{cost}bps"] = expected
                summary[name] = result
                rows.append({
                    "month": month, "symbol": symbol, "strategy": name,
                    "opportunities": len(group),
                    "trades": stats["trades"],
                    "coverage": stats["coverage"],
                    **{
                        f"net_{cost}bps_per_trade": stats[f"net_{cost}bps_per_trade"]
                        for cost in COSTS
                    },
                    **{
                        f"net_{cost}bps_per_opportunity": stats[
                            f"net_{cost}bps_per_opportunity"
                        ]
                        for cost in COSTS
                    },
                    "random_expected_net_5bps_per_trade": result.get(
                        "random_expected_net_5bps_per_trade"
                    ),
                    "edge_vs_random_5bps_per_trade": result.get(
                        "edge_vs_random_5bps_per_trade"
                    ),
                })
            grouped[month][symbol] = summary

    overall = {}
    for name in STRATEGIES:
        joined = pd.concat(overall_outcomes[name], ignore_index=True)
        stats = _score(joined, total_opportunities=all_n)
        if name in MODEL_SPECS:
            for cost in COSTS:
                random_sum = overall_expected[name][cost]
                stats[f"random_expected_net_{cost}bps_per_trade"] = (
                    float(random_sum/len(joined)) if len(joined) else None
                )
                stats[f"edge_vs_random_{cost}bps_per_trade"] = (
                    float(stats[f"net_{cost}bps_per_trade"] -
                          random_sum/len(joined)) if len(joined) else None
                )
        overall[name] = stats

    population = {
        "all_matured_events": int(len(events)),
        "train_matured_events": int(len(_partition(events, START, TRAIN_END))),
        "april_matured_events": int(len(april)),
        "may_june_matured_events": int(all_n),
        "may_june_B_confirmed": int(overall["B_confirmed"]["trades"]),
    }
    if census_by_symbol is not None:
        if set(census_by_symbol) != set(SYMBOLS):
            raise ValueError("Four symbol census required")
        for symbol in SYMBOLS:
            counts = census_by_symbol[symbol]
            if counts["matured_A"] != len(events_by_symbol[symbol]):
                raise AssertionError(f"{symbol}: maturity census discrepancy")
            if counts["confirmed_B"] != int(
                events_by_symbol[symbol].baseline_b_confirmed.sum()
            ):
                raise AssertionError(f"{symbol}: confirmation census discrepancy")
        population["by_asset_pre_split"] = census_by_symbol

    report = {
        "protocol": PROTOCOL,
        "status": "REUSED_HISTORY_EXPLORATORY_NOT_INDEPENDENT_OOS",
        "source_dataset_fingerprint_sha256": fingerprint,
        "symbols": list(SYMBOLS),
        "cutoff_exclusive_utc": CUTOFF.isoformat(),
        "split_boundaries": {
            "long_training_start": START.isoformat(),
            "short_training_start": SHORT_START.isoformat(),
            "training_end_exclusive": TRAIN_END.isoformat(),
            "score_calibration_end_exclusive": APRIL_END.isoformat(),
            "may_end_exclusive": MAY_END.isoformat(),
            "june_end_exclusive": CUTOFF.isoformat(),
        },
        "models": model_details,
        "population": population,
        "diagnostics": grouped,
        "combined_may_june": overall,
        "primary_metric": "net_5bps_per_trade compared to equal-count B random",
        "warning": (
            "Already-explored historical diagnostics; not an independent OOS "
            "or executable portfolio return. Hypothetical entry OPEN(t+2) "
            "at the exact confirmation close boundary; real slippage/spread "
            "and fills are unverified. No parameter search; never-trade=0."
        ),
    }
    return report, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P4 V3 preregistered 18-month Ridge comparison (EXPLORATORY)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    root, jpath, cpath = (
        args.data_root.resolve(), args.output_json.resolve(),
        args.output_csv.resolve(),
    )
    if (jpath == cpath or jpath.parent != cpath.parent
            or not jpath.parent.is_dir()
            or any(path.exists() or path.is_relative_to(root)
                   for path in (jpath, cpath))):
        parser.error("Require two NEW output files in same existing dir outside P0")

    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint

    candles, manifests = load_long_development_history(CandleStore(root))
    events = {}
    census = {}
    for symbol in SYMBOLS:
        events[symbol], census[symbol] = prepare_symbol(candles[symbol])
        s = census[symbol]
        print(f"{symbol}: signals={s['signals_unlabeled']} "
              f"matured_A={s['matured_A']} B={s['confirmed_B']}", flush=True)
    if sum(v["signals_unlabeled"] for v in census.values()) != 21583:
        # Prespecified operator-confirmed UNLABELED V3 count before outcomes.
        raise ValueError("Expected 21,583 V3 unlabeled signals; data drift")
    report, table = compare_fixed_models(
        events, manifest_fingerprint(manifests), census
    )
    with cpath.open("x", encoding="utf-8", newline="") as f:
        table.to_csv(f, index=False)
    with jpath.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Fixed thresholds (predicted net@5bps):")
    for name, details in report["models"].items():
        print(f"  {name}: {details['threshold_score_net_5bps']:.4f}; "
              f"train B={details['train_B_events']}")
    for name, stats in report["combined_may_june"].items():
        print(f"  {name}: trades={stats['trades']}, "
              f"net@5/trade={stats['net_5bps_per_trade']}, "
              f"net@5/opportunity={stats['net_5bps_per_opportunity']}, "
              f"edge-vs-matched-random/trade={stats.get('edge_vs_random_5bps_per_trade')}")
    print("REUSED HISTORICAL DIAGNOSTIC ONLY; NOT INDEPENDENT OOS; ZERO=0")
    print("Saved", jpath, cpath)


if __name__ == "__main__":
    main()
