"""Read-only P2 H2/H6 detection-error audit from EXISTING validation JSON.

No retraining, market data loading, sealed OOS, ADA, prospective scores,
feature selection or production side effects. Explains whether selected
directional predictions fail as NEUTRAL or the opposite direction.
The validation checks have already been inspected: descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.feature_reduction_diagnostics import BASELINE, HORIZONS, PROTOCOL as SOURCE_PROTOCOL

PROTOCOL = "p2_h2_h6_existing_check_error_audit_v1"
EXPECTED_FOLDS = 36


def _numerical(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Empty or non-finite metric distribution")
    return a


def _audit_metrics(cells: list[dict]) -> dict:
    """Aggregate check-only fields; do not confuse descriptive folds with iid tests."""
    if not cells:
        raise ValueError("No valid check metrics")
    for c in cells:
        required = ("selected", "n_total", "correct_direction_class",
                    "neutral_among_selected", "random_reference_precision",
                    "uplift_points", "coverage", "per_direction")
        if any(k not in c for k in required):
            raise ValueError("Incomplete detection metrics")
        if any(k not in c["per_direction"] for k in ("DOWN", "UP")):
            raise ValueError("Missing per-direction detection metrics")

    selected = sum(int(c["selected"]) for c in cells)
    total = sum(int(c["n_total"]) for c in cells)
    hits = sum(int(c["correct_direction_class"]) for c in cells)
    neutral = sum(int(c["neutral_among_selected"]) for c in cells)
    errors_opposite = selected - hits - neutral
    if (total <= 0 or selected <= 0 or
            min(hits, neutral, errors_opposite) < 0 or selected > total):
        raise ValueError("Inconsistent selected counts")
    direction = {}
    for side in ("DOWN", "UP"):
        predicted = sum(int(c["per_direction"][side]["predicted"]) for c in cells)
        correct = sum(int(c["per_direction"][side]["correct"]) for c in cells)
        true_support = sum(int(c["per_direction"][side]["support"]) for c in cells)
        if predicted < correct or true_support < correct:
            raise ValueError("Impossible confusion support")
        direction[side] = {
            "predicted": predicted,
            "correct": correct,
            "precision": correct / predicted if predicted else None,
            "recall_on_all_check_rows":correct / true_support if true_support else None,
            "truth_support":true_support,
        }
    if (sum(x["predicted"] for x in direction.values()) != selected or
            sum(x["correct"] for x in direction.values()) != hits):
        raise ValueError("Directional counts do not add up")

    expected_random_hits = sum(
        float(c["random_reference_precision"])*int(c["selected"])
        for c in cells
    )
    uplift = _numerical([c["uplift_points"] for c in cells])
    coverages = _numerical([c["coverage"] for c in cells])
    if any(abs(float(c["coverage"])-int(c["selected"])/int(c["n_total"]))>1e-8
           for c in cells):
        raise ValueError("Invalid coverage accounting")
    if any(abs(float(c["uplift_points"])-
               100*(int(c["correct_direction_class"])/int(c["selected"])-
                    float(c["random_reference_precision"])))>1e-6
           for c in cells):
        raise ValueError("Invalid per-fold uplift accounting")

    result = {
        "folds_or_asset_fold_records":len(cells),
        "check_rows":total,
        "selected":selected,
        "precision_selected":hits/selected,
        "coverage_realized":selected/total,
        "random_reference_selected_weighted":expected_random_hits/selected,
        "uplift_selected_weighted_points":100*(hits-expected_random_hits)/selected,
        "mean_fold_uplift_points":float(uplift.mean()),
        "median_fold_uplift_points":float(np.median(uplift)),
        "fold_uplift_q25_points":float(np.quantile(uplift,0.25)),
        "fold_uplift_q75_points":float(np.quantile(uplift,0.75)),
        "min_fold_uplift_points":float(uplift.min()),
        "max_fold_uplift_points":float(uplift.max()),
        "folds_above_reference":int((uplift>0).sum()),
        "folds_at_or_below_reference":int((uplift<=0).sum()),
        "mean_fold_coverage":float(coverages.mean()),
        "selected_correct":hits,
        "selected_truth_neutral":neutral,
        "selected_truth_opposite":errors_opposite,
        "fraction_of_selected_neutral_errors":neutral/selected,
        "fraction_of_selected_opposite_errors":errors_opposite/selected,
        "fraction_of_wrong_decisions_neutral":neutral/(selected-hits)
            if selected>hits else None,
        "directional":direction,
    }
    if all("raw_log_loss" in c and c["raw_log_loss"] is not None for c in cells):
        result["mean_fold_raw_log_loss"]=float(_numerical(
            [c["raw_log_loss"] for c in cells]).mean())
    return result


def audit(source:dict) -> dict:
    if (source.get("protocol")!=SOURCE_PROTOCOL or
            source.get("mode") not in ("group","individual") or
            source.get("folds_analyzed")!=EXPECTED_FOLDS or
            source.get("research_status")!="EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS"):
        raise ValueError("Expect the existing complete P2 check-only ablation report")
    for k in ("ADA_loaded_or_scored","fold_OOS_scored","final_90_days_scored",
              "frozen_P3_models_modified"):
        if source.get(k) is not False:
            raise ValueError(f"Sealed/private-data safety flag failed: {k}")
    rows = source.get("folds")
    if not isinstance(rows,list) or len(rows)!=EXPECTED_FOLDS:
        raise ValueError("Exactly 36 historical dev fold records required")
    if sorted(int(r["fold_id"]) for r in rows)!=list(range(EXPECTED_FOLDS)):
        raise ValueError("Bad or duplicated fold ids")
    report = {}
    for horizon in HORIZONS:
        key = f"H{horizon}"
        feasible = []
        infeasible = []
        for r in sorted(rows,key=lambda x:int(x["fold_id"])):
            result = r["horizons"][key][BASELINE]
            if result["feasible"]:
                if result.get("check_metrics") is None or not isinstance(
                    result.get("by_asset"),dict
                ):
                    raise ValueError("Feasible fold lacks baseline metrics")
                if set(result["by_asset"])!=set(SEEN_SYMBOLS):
                    raise ValueError("Each baseline check needs four seen assets")
                feasible.append((int(r["fold_id"]),result))
            else:
                infeasible.append(int(r["fold_id"]))
        if not feasible:
            raise ValueError("No feasible folds for "+key)
        pooled = _audit_metrics([r["check_metrics"] for _,r in feasible])
        by_asset = {
            symbol:_audit_metrics([r["by_asset"][symbol] for _,r in feasible])
            for symbol in SEEN_SYMBOLS
        }
        if sum(v["selected"] for v in by_asset.values())!=pooled["selected"]:
            raise ValueError("Four asset decisions do not match pooled count")
        if sum(v["selected_correct"] for v in by_asset.values())!=pooled["selected_correct"]:
            raise ValueError("Four asset correct decisions do not match pooled count")
        by_fold=[{
            "fold_id":fold_id,
            "uplift_points":float(outcome["check_metrics"]["uplift_points"]),
            "coverage":float(outcome["check_metrics"]["coverage"]),
            "precision":float(outcome["check_metrics"]["directional_precision"]),
            "selected_truth_neutral_rate":outcome["check_metrics"]["neutral_among_selected"]
                                        /outcome["check_metrics"]["selected"],
        } for fold_id,outcome in feasible]
        mid=len(by_fold)//2
        report[key]={
            "feasible_folds":len(feasible),
            "infeasible_fold_ids":infeasible,
            "overall":pooled,
            "by_asset":by_asset,
            "earlier_feasible_half_mean_uplift_points":float(
                np.mean([x["uplift_points"] for x in by_fold[:mid]])),
            "later_feasible_half_mean_uplift_points":float(
                np.mean([x["uplift_points"] for x in by_fold[mid:]])),
            "worst_five_fold_ids_by_uplift":[x["fold_id"] for x in sorted(
                by_fold,key=lambda x:x["uplift_points"])[:5]],
            "per_fold":by_fold,
        }
    return {
        "protocol":PROTOCOL,
        "source_protocol":SOURCE_PROTOCOL,
        "source_research_status":source["research_status"],
        "source_dataset_fingerprint_sha256":source.get("dataset_fingerprint_sha256"),
        "source_fold_count":EXPECTED_FOLDS,
        "no_new_training":True,
        "no_raw_candle_loading":True,
        "no_ada_oos_final90_scoring":True,
        "research_status":"DESCRIPTIVE_EXISTING_REPEATEDLY_INSPECTED_CHECKS",
        "explanation":[
            "Direction false positives split into true NEUTRAL versus true OPPOSITE.",
            "Selected-weighted pooled uplift and unweighted per-fold uplift are distinct.",
            "Assets share dates; folds overlap in training. No iid inference.",
            "This file identifies error types, not causal reasons for errors.",
            "No feature/hyperparameter selection, live-readiness, PnL or promotions.",
        ],
        "horizons":report,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Historical detection-error audit from completed JSON only")
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite existing detection audit")
    if args.source.resolve()==args.output.resolve():
        p.error("Source and destination cannot match")
    payload=args.source.read_bytes()
    result=audit(json.loads(payload))
    result["source_json_sha256"]=hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2,ensure_ascii=False,allow_nan=False)
    for horizon in HORIZONS:
        h=result["horizons"][f"H{horizon}"]
        m=h["overall"]
        print(
            f"H{horizon}: feasible={h['feasible_folds']}/{EXPECTED_FOLDS} "
            f"precision={100*m['precision_selected']:.2f}% "
            f"reference={100*m['random_reference_selected_weighted']:.2f}% "
            f"uplift_pooled={m['uplift_selected_weighted_points']:+.3f} pt "
            f"coverage={100*m['coverage_realized']:.2f}% "
            f"errors_neutral={100*m['fraction_of_selected_neutral_errors']:.2f}% "
            f"errors_opposite={100*m['fraction_of_selected_opposite_errors']:.2f}%",
            flush=True,
        )
    print("Saved descriptive error audit:",args.output.resolve(),flush=True)


if __name__=="__main__":
    main()
