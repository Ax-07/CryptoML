"""P3 PERSONAL V2: new prospective registration, research-only local evidence.

The original P3 V1 protocol and registration are NEVER edited, re-armed or
misrepresented as independent proof. Scientific hypotheses remain unchanged.
V2 expressly freezes local evidence mode, with no trusted external timestamp.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mlpipeline.prospective_protocol import (
    DAYS, EARLIEST_NEW_UTC, _canonical, _create_exclusive, _date, _iso,
    _sha, _utc, activate as _activate_v1, verify_activation as _verify_activation_v1,
    spec as _scientific_spec,
)
from mlpipeline.prospective_preflight import (
    EXPECTED_PACK_SHA256, EXPECTED_POLICY_SHA256, file_hash,
)

PROTOCOL_ID = "cryptoml_p3_h2_h6_personal_local_v2"
EVIDENCE_MODE = "personal_local_no_external_anchor"
MIN_NOTICE = timedelta(days=2)


def spec() -> dict:
    research = copy.deepcopy(_scientific_spec())
    research["protocol_id"] = PROTOCOL_ID
    research["research_phase"] = "personal_prospective_research_no_trading"
    research["evidence"] = {
        "mode": EVIDENCE_MODE,
        "external_timestamp": False,
        "local_sha256_chained_predictions": True,
        "local_sha256_chained_attempts": True,
        "no_backfill_after_failed_or_crashed_attempt": True,
        "no_independent_proof_of_antecedence_or_non_tampering": True,
    }
    research["limitations"].append(
        "Personal-only: device clock, local SHA256 and backups are not "
        "independent evidence of prediction time or integrity."
    )
    return research


def make_registration(*, start: datetime, now: datetime) -> dict:
    now, start = _utc(now), _utc(start)
    if start != start.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("Personal P3 start must be midnight UTC")
    if start < max(now + MIN_NOTICE, EARLIEST_NEW_UTC):
        raise ValueError("Personal P3 requires >=48h advance registration")
    end = start + timedelta(days=DAYS)
    return {
        "registered_at_utc": _iso(now),
        "prospective_start_utc": _iso(start),
        "prospective_end_exclusive_utc": _iso(end),
        "evaluation_not_before_utc": _iso(end + timedelta(minutes=60)),
        "status": "registered_only_not_active",
        "protocol": spec(),
        "activation_requirements": [
            "exact operator-pinned pre-July H2/H6 model and policy SHA256",
            "unchanged horizons, frozen thresholds and feature schema",
            "single PC operator: UTC clock synchronization and supervised M5 execution",
            "local SHA256 prediction/attempt chains + no retrospective prediction writes",
            "optional scheduled backups to an independent operator-chosen destination",
            "no early outcome scoring; final evaluation after 56d and 60min purge",
            "local evidence only: never claim independently verified timestamps",
        ],
    }


def register(root: Path, *, start: datetime, now: datetime | None = None) -> dict:
    if root.exists():
        raise FileExistsError("Personal P3 registration must use a NEW directory")
    pay = make_registration(start=start, now=now or datetime.now(timezone.utc))
    raw = _canonical(pay)
    root.mkdir(parents=True, exist_ok=False)
    _create_exclusive(root / "protocol.json", raw)
    _create_exclusive(root / "protocol.sha256", (_sha(raw) + "\n").encode("ascii"))
    return pay


def verify(root: Path) -> dict:
    raw = (root / "protocol.json").read_bytes()
    if _sha(raw) != (root / "protocol.sha256").read_text("ascii").strip():
        raise ValueError("Personal protocol SHA256 mismatch")
    pay = json.loads(raw)
    if _canonical(pay) != raw or pay.get("protocol") != spec():
        raise ValueError("Personal P3 frozen protocol changed")
    if pay.get("status") != "registered_only_not_active":
        raise ValueError("Personal P3 registration status changed")
    created = _date(pay["registered_at_utc"])
    start = _date(pay["prospective_start_utc"])
    end = _date(pay["prospective_end_exclusive_utc"])
    if (start != start.replace(hour=0, minute=0, second=0, microsecond=0)
            or start < max(created + MIN_NOTICE, EARLIEST_NEW_UTC)
            or end != start + timedelta(days=DAYS)
            or _date(pay["evaluation_not_before_utc"]) != end + timedelta(minutes=60)):
        raise ValueError("Invalid personal P3 calendar or minimum notice")
    return pay


def _pack_files(pack_root: Path) -> tuple[Path, Path, Path]:
    return (pack_root / "h2_model.joblib", pack_root / "h6_model.joblib",
            pack_root / "threshold_policy.json")


def verify_frozen_packs(pack_root: Path) -> tuple[Path, Path, Path]:
    h2, h6, policy = _pack_files(pack_root)
    for horizon, path in (("H2", h2), ("H6", h6)):
        if file_hash(path) != EXPECTED_PACK_SHA256[horizon]:
            raise ValueError(f"{horizon} differs from operator-pinned SHA256")
    if file_hash(policy) != EXPECTED_POLICY_SHA256:
        raise ValueError("Threshold policy differs from operator-pinned SHA256")
    return h2, h6, policy


def activate(root: Path, *, pack_root: Path, now: datetime | None = None) -> dict:
    """Operator-only freeze, requires exactly previously reviewed real packs."""
    verify(root)
    h2, h6, policy = verify_frozen_packs(pack_root)
    return _activate_v1(
        root, model_h2=h2, model_h6=h6, threshold_file=policy,
        now=now, verifier=verify,
    )


def verify_activation(root: Path) -> dict:
    frozen = _verify_activation_v1(root, verifier=verify)
    if frozen.get("protocol_sha256") != (root / "protocol.sha256").read_text("ascii").strip():
        raise ValueError("Personal P3 activation does not bind to V2 registration")
    if frozen.get("status") != "model_hashes_and_threshold_policy_locked_forecast_pipeline_not_verified":
        raise ValueError("Unexpected personal P3 activation state")
    for horizon in ("H2", "H6"):
        if frozen["artifact_files_sha256"][horizon] != EXPECTED_PACK_SHA256[horizon]:
            raise ValueError(f"{horizon} is not the operator-reviewed frozen pack")
    if frozen["threshold_policy_sha256"] != EXPECTED_POLICY_SHA256:
        raise ValueError("Frozen threshold policy fingerprint mismatch")
    return frozen


def main():
    parser = argparse.ArgumentParser(
        description="P3 PERSONAL V2 local-only registration/freezing (no scoring)"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    reg = actions.add_parser("register")
    reg.add_argument("--root", type=Path, required=True)
    reg.add_argument("--start-utc", required=True,
                     help="Explicit FUTURE YYYY-MM-DDT00:00:00Z; >=48h notice")
    for mode in ("verify", "activate", "verify-activation"):
        cmd = actions.add_parser(mode)
        cmd.add_argument("--root", type=Path, required=True)
        if mode == "activate":
            cmd.add_argument("--pack-root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        result = register(args.root, start=_date(args.start_utc))
        print("PERSONAL P3 V2 REGISTERED; NOT ACTIVE", result["prospective_start_utc"])
        print("Research-only local timestamps, not independently certified.")
    elif args.action == "verify":
        print("PERSONAL P3 V2 PROTOCOL SHA256 VERIFIED; NOT ACTIVATED",
              verify(args.root)["prospective_start_utc"])
    elif args.action == "activate":
        result = activate(args.root, pack_root=args.pack_root)
        print("MODEL HASHES LOCKED FOR PERSONAL P3 V2; scheduler NOT STARTED",
              result["activated_at_utc"])
    else:
        print("PERSONAL P3 V2 PACK HASHES VERIFIED; no predictions checked",
              verify_activation(args.root)["activated_at_utc"])


if __name__ == "__main__":
    main()
