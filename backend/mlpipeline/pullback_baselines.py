"""P4 exploratory fixed-rule pullback A/B comparison on PRE-JULY P0 data only.

A: every matured EMA9 pullback; fixed delayed-entry 1ATR stop/1.5ATR target.
B: same hypothetical fill and barriers, but take only pullbacks confirmed at
   the close of the immediately next candle (known before that fill).
No model training, model tuning, prospective evidence, or actual executable fills.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.pullback_frequency import CUTOFF, START, SYMBOLS
from mlpipeline.trend_pullback import M5

PROTOCOL = "trend_pullback_v1_fixed_A_vs_B_prejuly_exploratory"
COSTS = (2, 5, 10)


def annotate_confirmation(
    candles: pd.DataFrame, detected: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Mark B using ONLY t+1 bar, strictly before the common t+2 entry.

    Only already-matured A events from outcomes are eligible; B never gets
    hypothetical access to the same entry if t+1 is not actually closed.
    """
    required_outcome = {"signal_close_at", "entry_at", "mature_at", "gross_bps"}
    if (not required_outcome.issubset(outcomes.columns)
            or len(detected) != len(candles)
            or not detected.bar_close_at.equals(candles.bar_close_at)):
        raise ValueError("Detector/candle/outcome input mismatch")
    if not detected.bar_close_at.is_unique or not outcomes.signal_close_at.is_unique:
        raise ValueError("Duplicate signal timestamps")
    if not outcomes.empty and (not outcomes.mature_at.lt(CUTOFF).all()
                               or not outcomes.signal_close_at.ge(START).all()):
        raise ValueError("Outcomes out of P0 scope")
    index_by_time = {value: i for i, value in enumerate(detected.bar_close_at)}
    confirmed = []
    for item in outcomes.itertuples(index=False):
        i = index_by_time.get(item.signal_close_at)
        if i is None or not bool(detected.pullback_signal.iloc[i]):
            raise ValueError("Outcome outside predeclared pullback candidates")
        if (i + 2 >= len(candles)
                or candles.bar_close_at.iloc[i+1] != item.signal_close_at + M5
                or item.entry_at != candles.bar_open_at.iloc[i+2]
                or item.entry_at != item.signal_close_at + M5):
            raise ValueError("Inconsistent confirmation or entry time")
        # At close of t+1 this candle and its EMA are already known.
        next_bar = candles.iloc[i+1]
        next_ema9 = float(detected.ema9.iloc[i+1])
        confirmed.append(bool(
            np.isfinite(next_ema9)
            and next_bar.close > next_bar.open
            and next_bar.close > next_ema9
            and next_bar.close > candles.close.iloc[i]
        ))
    annotated = outcomes.copy()
    annotated["baseline_b_confirmed"] = confirmed
    return annotated


def _score(outcomes: pd.DataFrame, *, total_opportunities: int) -> dict:
    if total_opportunities < len(outcomes):
        raise ValueError("Trades exceed total opportunities")
    result = {"opportunities": total_opportunities,
              "trades": int(len(outcomes)),
              "coverage": len(outcomes) / total_opportunities if total_opportunities else None}
    for cost in COSTS:
        key = f"net_{cost}bps"
        net = outcomes[key].to_numpy(dtype=float)
        if not np.isfinite(net).all():
            raise ValueError("Nonfinite net hypothetical return")
        result[f"sum_net_{cost}bps"] = float(net.sum())
        result[f"net_{cost}bps_per_trade"] = float(net.mean()) if len(net) else None
        result[f"net_{cost}bps_per_opportunity"] = float(net.sum()/total_opportunities) if total_opportunities else None
        result[f"positive_net_{cost}bps_trades"] = int((net > 0).sum())
        result[f"positive_net_{cost}bps_trade_fraction"] = float((net > 0).mean()) if len(net) else None
    reasons = outcomes.exit_reason.value_counts().to_dict() if "exit_reason" in outcomes else {}
    result["exit_reasons"] = {str(k): int(v) for k, v in reasons.items()}
    return result


def _cohorts(events: pd.DataFrame) -> dict:
    a_count = len(events)
    b = events.loc[events.baseline_b_confirmed]
    a_score = _score(events, total_opportunities=a_count)
    b_score = _score(b, total_opportunities=a_count)
    return {
        "A_take_every_matured_pullback": a_score,
        "B_confirm_next_close_before_common_entry": b_score,
        "B_minus_A_net_2bps_per_opportunity": (
            b_score["net_2bps_per_opportunity"] - a_score["net_2bps_per_opportunity"]
            if a_count else None
        ),
    }


def summarize_prepared(annotated: dict[str, pd.DataFrame], fingerprint: str) -> tuple[dict, pd.DataFrame]:
    """Aggregate A/B across assets and UTC calendar months, without tuning."""
    if set(annotated) != set(SYMBOLS) or len(fingerprint) != 64:
        raise ValueError("Four seen symbols and dataset fingerprint required")
    by_asset = {}
    rows = []
    all_events = []
    for symbol in SYMBOLS:
        events = annotated[symbol].copy()
        if not {"signal_close_at", "baseline_b_confirmed", "net_2bps", "net_5bps", "net_10bps"}.issubset(events.columns):
            raise ValueError(f"Invalid annotated events: {symbol}")
        if not events.empty:
            if (not events.signal_close_at.ge(START).all()
                    or not events.signal_close_at.lt(CUTOFF).all()
                    or not events.signal_close_at.is_unique
                    or not events.signal_close_at.is_monotonic_increasing
                    or not pd.api.types.is_bool_dtype(events.baseline_b_confirmed)):
                raise ValueError(f"Invalid timestamps/confirmation flags: {symbol}")
        events["symbol"] = symbol
        events["month"] = (events.signal_close_at.dt.strftime("%Y-%m")
                           if not events.empty else pd.Series(index=events.index, dtype="string"))
        all_events.append(events)
        summary = _cohorts(events)
        by_month = {}
        for month, group in events.groupby("month", sort=True):
            by_month[month] = _cohorts(group)
            for baseline, result in (("A", by_month[month]["A_take_every_matured_pullback"]),
                                     ("B", by_month[month]["B_confirm_next_close_before_common_entry"])):
                rows.append({"symbol": symbol, "month": month, "baseline": baseline,
                             **{key: val for key, val in result.items() if key != "exit_reasons"}})
        for baseline in ("A_take_every_matured_pullback", "B_confirm_next_close_before_common_entry"):
            if sum(by_month[month][baseline]["trades"] for month in by_month) != summary[baseline]["trades"]:
                raise AssertionError(f"Monthly trade mismatch: {symbol}/{baseline}")
            for cost in COSTS:
                key = f"sum_net_{cost}bps"
                if not np.isclose(sum(by_month[m][baseline][key] for m in by_month),summary[baseline][key]):
                    raise AssertionError(f"Monthly sums mismatch: {symbol}/{baseline}/{cost}")
        by_asset[symbol] = {"overall": summary, "by_month": by_month}
    merged = pd.concat(all_events, ignore_index=True)
    report = {
        "protocol": PROTOCOL,
        "status": "EXPLORATORY_FIXED_BASELINES_PREJULY_NOT_INDEPENDENT_OOS",
        "source_dataset_fingerprint_sha256": fingerprint,
        "scope_start_inclusive_utc": START.isoformat(),
        "scope_end_exclusive_utc": CUTOFF.isoformat(),
        "scope": list(SYMBOLS),
        "costs_round_trip_bps": list(COSTS),
        "baseline_A": "all matured EMA9 first-touch signals; hypothetical t+2 open, ATR stop/target, 6 bars",
        "baseline_B": "A subset with t+1 close > open, t+1 close > EMA9(t+1), t+1 close > t close; same hypothetical t+2 open",
        "trade_overlap_note": "No aggregate portfolio accounting; gross and net bp sums are theoretical markouts.",
        "research_disclaimer": "Retrospectively reused history; not independent OOS, not prospective, no realistic fill/spread model, no training.",
        "overall": _cohorts(merged), "by_asset": by_asset,
    }
    return report, pd.DataFrame(rows)


def build_report(candles_by_symbol: dict, fingerprint: str) -> tuple[dict, pd.DataFrame]:
    if set(candles_by_symbol) != set(SYMBOLS):
        raise ValueError("Only four preapproved symbols may be evaluated")
    from mlpipeline.trend_pullback import detect_pullbacks
    from mlpipeline.pullback_outcomes import research_outcomes

    annotated = {}
    for symbol in SYMBOLS:
        candles = candles_by_symbol[symbol]
        if (candles.empty or candles.symbol.nunique() != 1 or candles.symbol.iloc[0] != symbol
                or not candles.bar_close_at.ge(START).all()
                or not candles.bar_close_at.lt(CUTOFF).all()):
            raise ValueError(f"Historical scope failure: {symbol}")
        detected = detect_pullbacks(candles)
        matured = research_outcomes(candles, development_end=CUTOFF)
        annotated[symbol] = annotate_confirmation(candles, detected, matured)
    return summarize_prepared(annotated, fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 fixed historical A/B baseline marks (not independent OOS)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    a = parser.parse_args()
    root = a.data_root.resolve()
    out_json, out_csv = a.output_json.resolve(), a.output_csv.resolve()
    if (out_json == out_csv or out_json.parent != out_csv.parent
            or not out_json.parent.is_dir()
            or any(x.exists() or x.is_relative_to(root) for x in (out_json,out_csv))):
        parser.error("Require two NEW separate result files in an existing folder outside P0")
    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint
    from mlpipeline.prospective_model_pack import load_clean_historic
    candles, manifests = load_clean_historic(CandleStore(root), verify_hashes=True)
    report, monthly = build_report(candles, manifest_fingerprint(manifests))
    # Save CSV before JSON; JSON marks completed run; never overwrite any old output.
    with out_csv.open("x", encoding="utf-8", newline="") as handle:
        monthly.to_csv(handle,index=False)
    with out_json.open("x", encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    for symbol, values in report["by_asset"].items():
        a_result = values["overall"]["A_take_every_matured_pullback"]
        b_result = values["overall"]["B_confirm_next_close_before_common_entry"]
        print(f"{symbol}: A={a_result['trades']}, B={b_result['trades']} "
              f"({100*b_result['coverage']:.1f}% confirms) "
              f"net@2bp/opportunity: A={a_result['net_2bps_per_opportunity']:.3f} "
              f"B={b_result['net_2bps_per_opportunity']:.3f}")
    print("HISTORICAL EXPLORATORY ONLY — no new independent evidence")
    print("Saved", out_json, out_csv)


if __name__ == "__main__":
    main()
