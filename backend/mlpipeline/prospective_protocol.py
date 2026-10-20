"""P3 prospective preregistration and model-freeze gates (NOT a backtest).

Does not score labels, read ADA, retrain models, or invent retrospective
forecasts. Online prediction collection and later scoring MUST be implemented
and verified separately; registration alone is NOT a prospective test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROTOCOL_ID = "cryptoml_p3_h2_h6_forward_confirmation_v1"
EARLIEST_NEW_UTC = datetime(2026, 10, 1, tzinfo=timezone.utc)
RESERVED_FINAL_WINDOW_START_UTC = datetime(2026, 7, 2, 23, 35, tzinfo=timezone.utc)
DAYS = 56
HORIZONS = (2, 6)
ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
FEATURE_SCHEMA = "features_v1"
LABEL_SCHEMA = "vol_scaled_v1"


def _utc(dt: datetime) -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        raise ValueError("UTC-aware datetime required")
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return _utc(dt).isoformat().replace("+00:00", "Z")


def _date(s: str) -> datetime:
    if not isinstance(s, str):
        raise ValueError("Invalid UTC timestamp")
    return _utc(datetime.fromisoformat(s.replace("Z", "+00:00")))


def _canonical(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec() -> dict:
    """Frozen research decisions: change implies a NEW versioned experiment."""
    return {
        "protocol_id": PROTOCOL_ID,
        "research_phase": "prospective_confirmation_not_live_trading",
        "symbols": list(ASSETS),
        "excluded_symbols": ["ADAUSDT"],
        "bar_timeframe": "M5_closed_only",
        "horizon_bars": list(HORIZONS),
        "prediction_horizon_minutes": {"H2": 10, "H6": 30},
        "features": {"schema": FEATURE_SCHEMA, "count": 30,
                     "groups_removed": []},
        "label": {"schema": LABEL_SCHEMA, "multiplier": 0.5,
                  "class_order": ["DOWN", "NEUTRAL", "UP"]},
        "model": {"type": "LightGBM_CPU_multiclass",
                  "n_estimators_max": 150, "seed": 42,
                  "class_prior_alpha": 1.0,
                  "architecture_modifications": False},
        "threshold": {
            "target_tuning_directional_coverage": 0.40,
            "tuning_source": "earlier_only_unlabeled_score_margins",
            "frozen_before_first_forward_prediction": True,
            "regime_specific_threshold_allowed_for_primary": True,
            "threshold_refit_during_confirmation": False,
        },
        "primary": {
            "cohort": "12_17_UTC_inclusive_six_hour_block",
            "selection": "H2_and_H6_both_at_40pct_tuning_target",
            "metric": "paired_weekly_mean_fold_local_directional_uplift_delta_points_H2_minus_H6",
            "comparison_unit": "8_consecutive_nonoverlapping_UTC_seven_day_blocks",
        },
        "secondary": [
            "whole_day_H2_vs_H6_at_same_40pct_tuning_target",
            "actual_check_coverage_H2_and_H6",
            "UP_DOWN_error_breakdown_per_asset",
            "raw_3class_logloss_skill_vs_frozen_training_prior",
        ],
        "evaluation": {
            "duration_days": DAYS,
            "minimum_expected_timestamp_coverage": 0.95,
            "minimum_realized_primary_coverage_per_horizon": 0.35,
            "maximum_realized_primary_coverage_per_horizon": 0.45,
            "maximum_absolute_primary_H2_H6_coverage_difference": 0.05,
            "minimum_positive_primary_weekly_deltas": 6,
            "minimum_positive_primary_mean_delta_points": 0.0,
            "interval_reporting": "weekly_block_bootstrap_descriptive_only",
            "single_final_scoring_pass": True,
            "horizon_maturity_delay_minutes": 60,
            "no_early_primary_result_peeking": True,
        },
        "immutability": {
            "no_retroactive_prediction_entry": True,
            "record_prediction_at_closed_M5_before_next_M5_close": True,
            "append_only_ledger_required": True,
            "two_models_and_thresholds_locked_before_start": True,
            "no_selection_using_new_outcomes": True,
            "no_trading_or_execution": True,
            "no_old_dev_or_reserved_historical_evaluation": True,
            "training_data_must_end_before_reserved_final_window": True,
        },
        "limitations": [
            "H2 and H6 label different future events; directional precision is not profitability.",
            "Hypothesis and 12-17 UTC cohort selected adaptively on previously inspected development checks.",
            "Weekly samples remain correlated; criteria are descriptive and not an independent p-value.",
            "Local timestamps and hashes cannot substitute for independently time-stamped append-only storage.",
            "Registration alone is NOT activation, online forecasting, or prospective validation.",
        ],
    }


def make_registration(*, now: datetime) -> dict:
    now = _utc(now)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # One full UTC day before collection to stage and freeze model packs.
    start = max(midnight + timedelta(days=2), EARLIEST_NEW_UTC)
    end = start + timedelta(days=DAYS)
    return {
        "registered_at_utc": _iso(now),
        "prospective_start_utc": _iso(start),
        "prospective_end_exclusive_utc": _iso(end),
        "evaluation_not_before_utc": _iso(end + timedelta(minutes=60)),
        "status": "registered_only_not_active",
        "protocol": spec(),
        "activation_requirements": [
            "trusted model packs for H2/H6 with 30-feature schema and train priors",
            "40pct tuning thresholds for both horizons locked BEFORE start",
            "verified model file SHA256 and reproducible training provenance",
            "append-only timestamped live forecast capture before H2 label matures",
            "independent external timestamp anchoring recommended",
            "no evaluation until entire 56d horizon and 60min maturity buffer end",
        ],
    }


def _create_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def register(root: Path, *, now: datetime | None = None) -> dict:
    if root.exists():
        raise FileExistsError("Prospective registration directory already exists")
    payload = make_registration(now=now or datetime.now(timezone.utc))
    raw = _canonical(payload)
    root.mkdir(parents=True, exist_ok=False)
    _create_exclusive(root / "protocol.json", raw)
    _create_exclusive(root / "protocol.sha256", (_sha(raw) + "\n").encode("ascii"))
    return payload


def verify(root: Path) -> dict:
    raw = (root / "protocol.json").read_bytes()
    expected = (root / "protocol.sha256").read_text("ascii").strip()
    if _sha(raw) != expected:
        raise ValueError("Protocol SHA256 mismatch: registration modified")
    payload = json.loads(raw)
    if _canonical(payload) != raw:
        raise ValueError("Protocol is not canonical JSON")
    if payload.get("protocol") != spec() or payload.get("status") != "registered_only_not_active":
        raise ValueError("Frozen protocol specification mismatch")
    created = _date(payload["registered_at_utc"])
    start = _date(payload["prospective_start_utc"])
    end = _date(payload["prospective_end_exclusive_utc"])
    allowed = _date(payload["evaluation_not_before_utc"])
    if (not created < start or start < EARLIEST_NEW_UTC
            or end != start + timedelta(days=DAYS)
            or allowed != end + timedelta(minutes=60)
            or start != start.replace(hour=0, minute=0, second=0, microsecond=0)):
        raise ValueError("Malformed prospective date bounds")
    return payload


def _validated_thresholds(obj: dict) -> dict:
    if set(obj) != {"H2", "H6"}:
        raise ValueError("Two horizons H2 and H6 required")
    for name in ("H2", "H6"):
        if set(obj[name]) != {"12-17_UTC", "all_day_UTC"}:
            raise ValueError("Require primary and secondary pre-tuned thresholds")
        for threshold in obj[name].values():
            if (type(threshold) not in (int, float)
                    or not math.isfinite(threshold)
                    or not 0 <= threshold <= 1):
                raise ValueError("Thresholds must be finite margins in [0,1]")
    return obj


def activate(root: Path, *, model_h2: Path, model_h6: Path,
             threshold_file: Path, now: datetime | None = None,
             verifier=None) -> dict:
    """Lock model hashes and policy; NO deserialization, inference or scoring.

    Caller must independently verify model provenance and an append-only
    live prediction pipeline. This gate is NOT an online evaluation.
    """
    payload = (verifier or verify)(root)
    now = _utc(now or datetime.now(timezone.utc))
    if now >= _date(payload["prospective_start_utc"]):
        raise ValueError("Too late to freeze models: prospective start reached")
    if (root / "activation.json").exists():
        raise FileExistsError("Activation already exists, refuses re-arm")
    if model_h2.resolve() == model_h6.resolve():
        raise ValueError("H2/H6 must use two distinct model artifacts")
    raw_models = {"H2": model_h2.read_bytes(), "H6": model_h6.read_bytes()}
    if any(not raw for raw in raw_models.values()):
        raise ValueError("Empty model artifact")
    policy_bytes = threshold_file.read_bytes()
    policy = json.loads(policy_bytes)
    if set(policy) != {"thresholds", "fitted_before_utc", "training_provenance"}:
        raise ValueError("Threshold policy needs provenance, fit date and thresholds")
    _validated_thresholds(policy["thresholds"])
    if _date(policy["fitted_before_utc"]) > now:
        raise ValueError("Threshold fitting date after activation")
    provenance = policy["training_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"H2", "H6"}:
        raise ValueError("Training provenance for both horizons is required")
    for h, horizon in (("H2", 2), ("H6", 6)):
        p = provenance[h]
        if (p.get("horizon_bars") != horizon
                or p.get("feature_schema") != FEATURE_SCHEMA
                or p.get("feature_count") != 30
                or p.get("label_schema") != LABEL_SCHEMA
                or p.get("alpha") != 1.0
                or not isinstance(p.get("train_prior_counts"), list)
                or len(p["train_prior_counts"]) != 3
                or any(type(x) is not int or x < 1 for x in p["train_prior_counts"])):
            raise ValueError(f"Invalid {h} immutable training metadata")
        if _date(p["training_data_last_close_utc"]) >= RESERVED_FINAL_WINDOW_START_UTC:
            raise ValueError("Training must exclude the reserved final 90-day window")
    activation = {
        "protocol_sha256": (root / "protocol.sha256").read_text("ascii").strip(),
        "activated_at_utc": _iso(now),
        "status": "model_hashes_and_threshold_policy_locked_forecast_pipeline_not_verified",
        "artifact_files_sha256": {h: _sha(raw) for h, raw in raw_models.items()},
        "artifact_paths_at_activation": {
            "H2": str(model_h2.resolve()), "H6": str(model_h6.resolve()),
        },
        "threshold_policy_sha256": _sha(policy_bytes),
        "threshold_policy": policy,
        "warnings": [
            "This locks hashes but cannot certify model training implementation or honest forecast timestamps.",
            "Independent event-time ledger and scoring remain prerequisites.",
        ],
    }
    raw = _canonical(activation)
    _create_exclusive(root / "activation.json", raw)
    _create_exclusive(root / "activation.sha256", (_sha(raw) + "\n").encode("ascii"))
    return activation


def verify_activation(root: Path, *, verifier=None) -> dict:
    registration = (verifier or verify)(root)
    raw = (root / "activation.json").read_bytes()
    expected = (root / "activation.sha256").read_text("ascii").strip()
    if _sha(raw) != expected:
        raise ValueError("Activation SHA256 mismatch")
    activation = json.loads(raw)
    if activation.get("protocol_sha256") != (root / "protocol.sha256").read_text("ascii").strip():
        raise ValueError("Activation does not match prospective protocol")
    if _date(activation["activated_at_utc"]) >= _date(registration["prospective_start_utc"]):
        raise ValueError("Model freeze occurred after collection start")
    if _canonical(activation) != raw:
        raise ValueError("Noncanonical activation JSON")
    _validated_thresholds(activation["threshold_policy"]["thresholds"])
    for h in ("H2", "H6"):
        path = Path(activation["artifact_paths_at_activation"][h])
        if not path.is_file() or _sha(path.read_bytes()) != activation["artifact_files_sha256"][h]:
            raise ValueError(f"Model pack {h} missing or modified")
    return activation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CryptoML P3 prospective preregistration (NO evaluation)"
    )
    modes = parser.add_subparsers(dest="command", required=True)
    for mode in ("register", "verify", "activate", "verify-activation"):
        cmd = modes.add_parser(mode)
        cmd.add_argument("--root", type=Path, required=True)
        if mode == "activate":
            cmd.add_argument("--h2-model", type=Path, required=True)
            cmd.add_argument("--h6-model", type=Path, required=True)
            cmd.add_argument("--threshold-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        p = register(args.root)
        print(f"Registered-only (NOT ACTIVE): {args.root.resolve()}")
        print(f"Collection start: {p['prospective_start_utc']}")
        print(f"Final evaluation not before: {p['evaluation_not_before_utc']}")
    elif args.command == "verify":
        p = verify(args.root)
        print(f"SHA256 OK: {p['protocol']['protocol_id']} (NOT ACTIVE)")
    elif args.command == "activate":
        activate(args.root, model_h2=args.h2_model, model_h6=args.h6_model,
                 threshold_file=args.threshold_file)
        print("Hashes locked, but live forecast capture remains unverified.")
    else:
        verify_activation(args.root)
        print("Protocol and pack fingerprints OK; NOT a prospective score.")


if __name__ == "__main__":
    main()
