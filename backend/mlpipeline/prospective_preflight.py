"""Read-only P3 preflight using the REAL frozen H2/H6 model packs.

No scoring, no P0 historical loading, no journal writes and no activation.
The operator-supplied SHA256 values are pinned from the 2026-10-14 pack
preparation transcript; a changed pack needs explicit independent review,
not a quiet acceptance of newly recomputed hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.prospective_live_capture import (
    _load_packs, fetch_latest_closed, latest_common_features, make_forecasts,
)
from mlpipeline.prospective_protocol import (
    ASSETS, _date, _sha, verify, RESERVED_FINAL_WINDOW_START_UTC,
)

# Checked independently by the operator after fitting; never change quietly.
EXPECTED_PACK_SHA256 = {
    "H2": "6dc2fc0a176bf2d719c7b468626bde2cebf16848d345e86268c4c1dc88a0f287",
    "H6": "d2745de33f21d8b1e64c7b2ad850f96a418e506352e03f92f394d7c041b9f097",
}
EXPECTED_POLICY_SHA256 = (
    "3dd655df3823fe9ddb f7ba6b1e721ed67981b90651c1c2ff16a6459f420a065a".replace(" ", "")
)
HISTORICAL_CUTOFF = _date("2026-07-01T00:00:00Z")


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _assert_provenance(policy: dict, *, start: datetime, now: datetime):
    if set(policy) != {"thresholds", "fitted_before_utc", "training_provenance"}:
        raise ValueError("Frozen threshold policy schema mismatch")
    if not _date(policy["fitted_before_utc"]) < min(start, now):
        raise ValueError("Frozen thresholds not fitted before registration start and now")
    if set(policy["training_provenance"]) != {"H2", "H6"}:
        raise ValueError("Missing exact H2/H6 training provenance")
    from mlpipeline.prospective_protocol import _validated_thresholds
    _validated_thresholds(policy["thresholds"])
    shared=[]
    for h in (2,6):
        key=f"H{h}"
        p=policy["training_provenance"][key]
        if (p.get("horizon_bars")!=h
                or p.get("feature_schema")!=FEATURE_SCHEMA_VERSION
                or p.get("label_schema")!=LABEL_SCHEMA_VERSION
                or p.get("feature_count")!=len(FEATURE_COLUMNS)
                or p.get("feature_columns")!=list(FEATURE_COLUMNS)
                or p.get("alpha")!=1.0):
            raise ValueError(f"{key}: provenance does not match frozen P3 schema")
        counts=p.get("train_prior_counts")
        if (not isinstance(counts,list) or len(counts)!=3
                or any(type(v) is not int or v<=0 for v in counts)):
            raise ValueError(f"{key}: invalid train priors")
        train=_date(p["training_data_last_close_utc"])
        stop=_date(p["early_stopping_last_close_utc"])
        tune=_date(p["tuning_last_close_utc"])
        historical=_date(p["historical_partition_cutoff_exclusive_utc"])
        if (not train < stop < tune < HISTORICAL_CUTOFF
                or historical != HISTORICAL_CUTOFF
                or train >= RESERVED_FINAL_WINDOW_START_UTC):
            raise ValueError(f"{key}: historical chronology violated")
        shared.append((train,stop,tune,historical,
                       p.get("dataset_fingerprint_sha256")))
    if shared[0]!=shared[1]:
        raise ValueError("H2 and H6 model packs use different training/tuning periods")


def validate_packs(pack_root: Path, protocol_root: Path, *, now=None) -> dict:
    """Verify actual immutable bytes before ANY joblib deserialization."""
    now=now or datetime.now(timezone.utc)
    protocol=verify(protocol_root)
    start=_date(protocol["prospective_start_utc"])
    files={
        "H2":pack_root/"h2_model.joblib",
        "H6":pack_root/"h6_model.joblib",
        "policy":pack_root/"threshold_policy.json",
    }
    hashes={key:file_hash(path) for key,path in files.items()}
    for h in ("H2","H6"):
        if hashes[h]!=EXPECTED_PACK_SHA256[h]:
            raise ValueError(f"{h}: model file digest differs from operator-frozen SHA256")
    if hashes["policy"]!=EXPECTED_POLICY_SHA256:
        raise ValueError("Threshold policy digest differs from operator-frozen SHA256")
    policy=json.loads(files["policy"].read_text(encoding="utf-8"))
    _assert_provenance(policy,start=start,now=now)
    # Allow only operator-reviewed, hash-pinned local joblib deserialization.
    activation={
        "artifact_paths_at_activation":{
            h:str(files[h].resolve()) for h in ("H2","H6")
        },
        "threshold_policy":policy,
    }
    packs=_load_packs(activation)
    for h in ("H2","H6"):
        obj=packs[h]
        expected=policy["training_provenance"][h]
        model=obj["estimator"]
        if (obj["features"]!=list(FEATURE_COLUMNS)
                or obj["train_prior_counts"]!=expected["train_prior_counts"]
                or obj["dataset_fingerprint_sha256"]!=expected["dataset_fingerprint_sha256"]
                or getattr(model,"n_features_in_",None)!=len(FEATURE_COLUMNS)
                or getattr(model,"best_iteration_",None)!=expected["model_best_iteration"]):
            raise ValueError(f"{h}: trained model contents do not match frozen provenance")
    return {
        "protocol":protocol,
        "policy":policy,
        "hashes":hashes,
        "packs":packs,
    }


def live_smoke_check(validated: dict, *, now=None, fetcher=fetch_latest_closed):
    """Use live CLOSED Binance candles, no future labels or saved forecasts."""
    now=now or datetime.now(timezone.utc)
    candles={symbol:fetcher(symbol,now=now) for symbol in ASSETS}
    rows,close=latest_common_features(candles,now=datetime.now(timezone.utc))
    # A read-only synthetic activation: this is NOT protocol activation.
    activation={"threshold_policy":validated["policy"]}
    outputs=make_forecasts(rows,validated["packs"],activation)
    for h in ("H2","H6"):
        if set(outputs[h])!=set(ASSETS):
            raise AssertionError("Missing asset predictions")
        for sym in ASSETS:
            item=outputs[h][sym]
            probs=np.asarray(item["raw_probabilities_down_neutral_up"],dtype=float)
            if (probs.shape!=(3,) or not np.isfinite(probs).all()
                    or (probs<0).any() or (probs>1).any()
                    or not np.isclose(probs.sum(),1.0,atol=1e-4)):
                raise ValueError(f"{h} {sym}: invalid multiclass probabilities")
            if not 0 <= item["margin"] <= 1:
                raise ValueError(f"{h} {sym}: invalid margin")
            if set(item["policies"])!={"12-17_UTC","all_day_UTC"}:
                raise ValueError(f"{h} {sym}: missing frozen policy")
            for decision in item["policies"].values():
                if decision["selected_direction"] not in (None,0,2):
                    raise ValueError(f"{h} {sym}: invalid selected direction")
    return {"bar_close_utc":close.isoformat(),
            "observed_assets":len(ASSETS),
            "horizons":list(outputs),
            "saved_or_anchored":False}


def main():
    p=argparse.ArgumentParser(description="P3 real-pack read-only readiness (NOT activation)")
    p.add_argument("--protocol-root",required=True,type=Path)
    p.add_argument("--pack-root",required=True,type=Path)
    p.add_argument("--live-check",action="store_true",
                   help="Query real closed Binance M5; never append or score")
    args=p.parse_args()
    valid=validate_packs(args.pack_root,args.protocol_root)
    print("SHA256 frozen packs H2/H6 + threshold policy: MATCH",flush=True)
    for h in ("H2","H6"):
        p=valid["policy"]["training_provenance"][h]
        print(f"{h}: best_iteration={p['model_best_iteration']}; "
              f"last_train={p['training_data_last_close_utc']}; "
              f"last_tune={p['tuning_last_close_utc']}",flush=True)
    if args.live_check:
        smoke=live_smoke_check(valid)
        print(f"Real-pack live-only inference OK: {smoke['bar_close_utc']}; "
              "4 assets, 2 horizons, both policies. NO persistence.",flush=True)
    else:
        print("Live inference NOT CHECKED (pass --live-check)",flush=True)
    print("NOT READY FOR P3 ACTIVATION: independent external anchor + "
          "continuous supervised capture unverified.",flush=True)


if __name__=="__main__":
    main()
