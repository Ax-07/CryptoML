"""P2 fixed-pair disagreement diagnostic, previously inspected dev checks ONLY.

Existing winner-by-mean (ADAPTIVELY observed) comparisons:
H2 LightGBM-30 vs reversion_3; H6 LightGBM-30 vs reversion_6.
Records only aggregated counters: agreement correct/neutral/opposite,
disagreement ML correct/reversion correct/both wrong neutral.
NO row predictions, new strategy, PnL, feature/threshold selection or promotion.

Replays exactly the existing 36-fold experiment, clipping BEFORE labeling
at EACH fold's own check end; H12-purged train120/early7/tune3/check4.
The frozen original rule-benchmark report MUST reproduce at each fold
(selected counts + correct counts); otherwise refuse to save the result.
Nothing reads or scores ADA, the current fold OOS, or final 90-day holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.directional_rule_benchmark import (
    HORIZONS, TARGET_COVERAGE, PROTOCOL as BASELINE_PROTOCOL,
    build_causal_contexts, prepare_fold_examples, parts_for_fold,
    rule_predictions,
)
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS, feature_matrix
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.predictability_decomposition import (
    REGIMES, regime_codes, training_regime_thresholds,
)
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL = "p2_h2_h6_ml_vs_reversion_disagreement_v1"
PAIRS = {2: "reversion_3", 6: "reversion_6"}
COUNT_KEYS = (
    "agree_correct",
    "agree_wrong_neutral",
    "agree_wrong_opposite",
    "disagree_ml_correct",
    "disagree_reversion_correct",
    "disagree_both_wrong_neutral",
)


def disagreement_counts(truth: np.ndarray, ml: np.ndarray, reversion: np.ndarray,
                        mask: np.ndarray) -> dict | None:
    """Exhaustive exclusive 6-way partition. Labels used ONLY for reporting."""
    y = np.asarray(truth)
    p = np.asarray(ml)
    r = np.asarray(reversion)
    chosen = np.asarray(mask)
    if (y.ndim != 1 or not len(y)
            or any(v.shape != y.shape for v in (p, r, chosen))
            or not np.isin(y, (0, 1, 2)).all()
            or not np.isin(p, (0, 2)).all()
            or not np.isin(r, (0, 2)).all()
            or chosen.dtype.kind != "b"):
        raise ValueError("Expected aligned direction labels/predictions and boolean mask")
    if not chosen.any():
        return None
    y, p, r = y[chosen], p[chosen], r[chosen]
    agree = p == r
    neutral = y == 1
    pmatch = p == y
    rmatch = r == y
    cells = {
        "agree_correct":int((agree & pmatch).sum()),
        "agree_wrong_neutral":int((agree & neutral).sum()),
        "agree_wrong_opposite":int((agree & ~neutral & ~pmatch).sum()),
        "disagree_ml_correct":int((~agree & pmatch).sum()),
        "disagree_reversion_correct":int((~agree & rmatch).sum()),
        "disagree_both_wrong_neutral":int((~agree & neutral).sum()),
    }
    n = len(y)
    if (sum(cells.values()) != n
            or (cells["disagree_ml_correct"] + cells["disagree_reversion_correct"]
                + cells["disagree_both_wrong_neutral"] != int((~agree).sum()))
            or (cells["agree_correct"] + cells["disagree_ml_correct"]
                != int(pmatch.sum()))
            or (cells["agree_correct"] + cells["disagree_reversion_correct"]
                != int(rmatch.sum()))):
        raise AssertionError("Incomplete or inconsistent disagreement partition")
    return {**cells, "n":int(n)}


def summarize_cells(cells: list[dict | None]) -> dict:
    """Recompute every derived metric from counts (not rounded per-fold rates)."""
    valid = [c for c in cells if c is not None]
    pooled = {k:sum(int(c[k]) for c in valid) for k in COUNT_KEYS}
    n = sum(int(c["n"]) for c in valid)
    if any(sum(int(c[k]) for k in COUNT_KEYS) != int(c["n"]) for c in valid):
        raise ValueError("Malformed fold disagreement cells")
    if sum(pooled.values()) != n:
        raise AssertionError("Aggregate count mismatch")
    agree = sum(pooled[k] for k in COUNT_KEYS if k.startswith("agree_"))
    # "disagree_" starts with different prefix; no overlap with "agree_".
    disagree = n - agree
    ml_correct = pooled["agree_correct"] + pooled["disagree_ml_correct"]
    rev_correct = pooled["agree_correct"] + pooled["disagree_reversion_correct"]
    mover_disagreements = (pooled["disagree_ml_correct"] +
                            pooled["disagree_reversion_correct"])
    differences = [
        100*(c["disagree_ml_correct"]-c["disagree_reversion_correct"])/c["n"]
        for c in valid
    ]
    return {
        "fold_records_with_rows":len(valid),
        "rows":n,
        "counts":pooled,
        "agreement_rate":agree/n if n else None,
        "disagreement_rate":disagree/n if n else None,
        "ml_precision":ml_correct/n if n else None,
        "reversion_precision":rev_correct/n if n else None,
        "net_correct_ml_minus_reversion":ml_correct-rev_correct,
        "net_gain_ml_points":100*(ml_correct-rev_correct)/n if n else None,
        "mean_fold_gain_ml_points":float(np.mean(differences)) if valid else None,
        "folds_ml_wins":sum(x>0 for x in differences),
        "folds_reversion_wins":sum(x<0 for x in differences),
        "folds_tied":sum(x==0 for x in differences),
        "disagreement_neutral_rate":(
            pooled["disagree_both_wrong_neutral"]/disagree
            if disagree else None),
        "ml_win_fraction_among_DISAGREEMENTS_with_true_movement":(
            pooled["disagree_ml_correct"]/mover_disagreements
            if mover_disagreements else None),
        "warning":"Matured check truth used only for retrospective evaluation; "
                  "folds/strategies not independent or actionable by themselves.",
    }


def verify_baseline_report(report: dict, source: dict) -> dict[int, dict]:
    if (report.get("protocol") != BASELINE_PROTOCOL
            or report.get("folds_analyzed") != 36
            or report.get("n_estimators") != 150
            or report.get("seed") != 42
            or report.get("ml_tuning_coverage_target") != TARGET_COVERAGE
            or report.get("source_dataset_fingerprint_sha256")
            != source.get("dataset_fingerprint_sha256")):
        raise ValueError("Expected fixed 36-fold 30-feature benchmark baseline")
    for key in ("ADA_loaded_or_scored","fold_OOS_test_labels_accessed",
                "final90_test_labels_accessed","P3_or_forward_models_modified"):
        if report.get(key) is not False:
            raise ValueError("Unsafe benchmark report: "+key)
    records = report.get("folds")
    if not isinstance(records,list) or len(records)!=36:
        raise ValueError("Benchmark needs 36 fold records")
    out={}
    for r in records:
        fid=r.get("fold_id")
        if type(fid) is not int or fid in out or not 0<=fid<36:
            raise ValueError("Bad benchmark fold id")
        out[fid]=r
    return out


def crosscheck_counts(existing:dict, observed:dict, pair_name:str)->None:
    """Reject even 1-row drift from previously completed benchmark."""
    for view in ("all_check","shared_ml_selected"):
        expected_rules=existing["rules"]
        present=observed[view]
        exp_ml=expected_rules["lightgbm_30"][view]
        exp_rev=expected_rules[pair_name][view]
        if present is None:
            if exp_ml is not None or exp_rev is not None:
                raise ValueError("Infeasible mask does not reproduce original")
        elif (exp_ml is None or exp_rev is None
              or present["n"] != exp_ml["n"]
              or present["n"] != exp_rev["n"]
              or present["agree_correct"]+present["disagree_ml_correct"]
                 != exp_ml["correct"]
              or present["agree_correct"]+present["disagree_reversion_correct"]
                 != exp_rev["correct"]):
            raise ValueError(f"{view}: row count/ML/reversion correct count drift")


def run_audit(candles:dict,source:dict,baseline:dict,*,
              fingerprint:str,max_folds:int|None=None)->dict:
    plan=load_plan(source)
    existing=verify_baseline_report(baseline,source)
    if fingerprint!=source.get("dataset_fingerprint_sha256"):
        raise ValueError("P0 dataset fingerprint differs from old P2 report")
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Exactly four allowed seen symbols, ADA forbidden")
    if max_folds is not None:
        if type(max_folds) is not int or not 1<=max_folds<=36:
            raise ValueError("Invalid max-folds")
        plan=plan[:max_folds]
    contexts=build_causal_contexts(candles,plan[-1]["check_end"])
    fold_results=[]
    for fold in plan:
        # Uses ONLY history strictly before THIS fold's check_end to label.
        shared=prepare_fold_examples(contexts,fold["check_end"])
        parts,bounds=parts_for_fold(
            shared,fold,source["folds"][fold["fold_id"]])
        train,early,tune,check=(parts[k] for k in (
            "train","early_stopping","tuning","validation_check"))
        sym=check.symbol.to_numpy()
        regime_bounds=training_regime_thresholds(train)
        regimes=regime_codes(check,regime_bounds)
        by_h={}
        for h,rev_name in PAIRS.items():
            key=f"H{h}"
            target=f"target_h{h}"
            ytrain=train[target].to_numpy(dtype=int)
            yearly=early[target].to_numpy(dtype=int)
            truth=check[target].to_numpy(dtype=int)
            prior_counts=np.bincount(ytrain,minlength=3)
            if (np.any(prior_counts==0)
                    or set(yearly.tolist())!={0,1,2}):
                raise ValueError("Unexpected 3-class training partition")
            model=_cpu_model(binary=False,n_estimators=150,seed=42)
            model.fit(
                feature_matrix(train),ytrain,
                eval_set=[(feature_matrix(early),yearly)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
            probs_t=_class_probs(model,feature_matrix(tune))
            pred_t,margin_t=decision_and_margin(
                probs_t,prior_counts,alpha=1.0)
            threshold=threshold_for_target(
                pred_t,margin_t,target_coverage=TARGET_COVERAGE)
            probs_c=_class_probs(model,feature_matrix(check))
            pred_c,margin_c=decision_and_margin(
                probs_c,prior_counts,alpha=1.0)
            selected = ((pred_c!=1)&(margin_c>=threshold["threshold"])
                        if threshold["feasible"]
                        else np.zeros(len(check),dtype=bool))
            rule_signs=rule_predictions(check,ytrain,probs_c)
            ml=rule_signs["lightgbm_30"]
            rev=rule_signs[rev_name]
            outputs={
                "all_check":disagreement_counts(
                    truth,ml,rev,np.ones(len(check),dtype=bool)),
                "shared_ml_selected":(
                    disagreement_counts(truth,ml,rev,selected)
                    if threshold["feasible"] else None),
            }
            earlier=existing[fold["fold_id"]]["horizons"][key]
            if threshold["feasible"] != earlier["ml_tuning"]["feasible"]:
                raise ValueError("Fold feasibility disagrees with old baseline")
            crosscheck_counts(earlier,outputs,rev_name)
            asset={}
            asset_regime={}
            for symbol in SEEN_SYMBOLS:
                symmask=sym==symbol
                asset[symbol]=(
                    disagreement_counts(truth,ml,rev,selected&symmask)
                    if threshold["feasible"] else None)
                asset_regime[symbol]={}
                for idx,name in enumerate(REGIMES):
                    asset_regime[symbol][name]=(
                        disagreement_counts(truth,ml,rev,
                            selected&symmask&(regimes==idx))
                        if threshold["feasible"] else None)
                if asset[symbol] is not None:
                    for key_count in ("n",*COUNT_KEYS):
                        if sum(
                            item[key_count] for item in
                            asset_regime[symbol].values() if item is not None
                        ) != asset[symbol][key_count]:
                            raise AssertionError("Regime slices do not sum to asset")
                expected_asset=earlier["by_asset_selected"][symbol]
                current_asset=asset[symbol]
                if current_asset is None:
                    if expected_asset["lightgbm_30"] is not None:
                        raise ValueError("Per-asset feasibility drift")
                elif (expected_asset["lightgbm_30"] is None
                      or current_asset["n"]!=expected_asset[
                          "lightgbm_30"]["n"]
                      or current_asset["agree_correct"]+
                          current_asset["disagree_ml_correct"]!=
                          expected_asset["lightgbm_30"]["correct"]
                      or current_asset["agree_correct"]+
                          current_asset["disagree_reversion_correct"]!=
                          expected_asset[rev_name]["correct"]):
                    raise ValueError("Per-asset baseline counts differ")
            whole=outputs["shared_ml_selected"]
            if whole is not None:
                for k in ("n",*COUNT_KEYS):
                    if sum(asset[s][k] for s in SEEN_SYMBOLS
                           if asset[s] is not None) != whole[k]:
                        raise AssertionError("Four assets do not sum to total")
            by_h[f"H{h}"]={
                "comparison_rule":rev_name,
                "ml_mask_feasible":threshold["feasible"],
                "check_coverage":float(selected.mean()) if threshold["feasible"] else None,
                **outputs,
                "by_asset_selected":asset,
                "by_asset_regime_selected":asset_regime,
                "train_regime_volatility_cutoffs":regime_bounds,
            }
            sel=outputs["shared_ml_selected"]
            print(
                f"fold={fold['fold_id']} H{h} "
                f"feasible={threshold['feasible']} "
                f"selected_n={0 if sel is None else sel['n']} "
                f"ml_wins={0 if sel is None else sel['disagree_ml_correct']} "
                f"reversion_wins={0 if sel is None else sel['disagree_reversion_correct']}",
                flush=True)
        fold_results.append({
            "fold_id":fold["fold_id"],"bounds":bounds,
            "horizons":by_h,
        })
        del shared,parts
    summaries={}
    for h in HORIZONS:
        key=f"H{h}"
        entries=[f["horizons"][key] for f in fold_results]
        summaries[key]={
            "comparison_rule":PAIRS[h],
            "all_check":summarize_cells([e["all_check"] for e in entries]),
            "shared_ml_selected":summarize_cells([
                e["shared_ml_selected"] for e in entries]),
            "by_asset_selected":{
                sym:summarize_cells([
                    e["by_asset_selected"][sym] for e in entries])
                for sym in SEEN_SYMBOLS
            },
            "by_asset_regime_selected":{
                sym:{
                    name:summarize_cells([
                        e["by_asset_regime_selected"][sym][name]
                        for e in entries])
                    for name in REGIMES
                } for sym in SEEN_SYMBOLS
            },
        }
    return {
        "protocol":PROTOCOL,
        "research_status":"DESCRIPTIVE_REPLAY_OF_REPEATEDLY_INSPECTED_DEV_CHECKS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "folds_analyzed":len(fold_results),
        "pairs_adaptively_chosen_from_prior_dev_checks":{
            f"H{h}":name for h,name in PAIRS.items()},
        "label_threshold_fixed":0.5,
        "n_estimators":150,"seed":42,
        "feature_count":len(FEATURE_COLUMNS),
        "target_tuning_coverage":TARGET_COVERAGE,
        "models_fit_per_fold":len(HORIZONS),
        "reproduced_original_benchmark_counts_each_fold":True,
        "no_row_predictions_saved":True,
        "ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,
        "final90_scored":False,
        "frozen_P3_or_forward_models_modified":False,
        "limitations":[
            "Existing checked windows and adaptive choice of reversion_3 vs reversion_6 "
            "make this a descriptive exploratory experiment, not confirmation.",
            "The selection mask is produced by LightGBM from unlabeled prior "
            "tuning scores, not by a standalone reversion strategy.",
            "ALL selected denominators are identical for both direction rules.",
            "Neutral true outcomes are errors for both binary directions.",
            "Volatility quantiles are computed on train rows separately by asset.",
            "Folds share assets/time and use overlapping training; no iid inference.",
            "No row-level predictions stored, no model promotion, PnL or costs.",
        ],
        "summary":summaries,
        "folds":fold_results,
    }


def main()->None:
    p=argparse.ArgumentParser(
        description="P2 descriptive ML versus reversion H2/H6 disagreement")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--fold-plan",type=Path,required=True)
    p.add_argument("--baseline-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--verify-hashes",action="store_true")
    args=p.parse_args()
    if args.months!=24:
        p.error("Exactly 24 months required by original P2 protocol")
    if args.output.exists():
        p.error("Refusing to overwrite existing diagnostic")
    if (args.output.resolve()==args.data_root.resolve()
            or args.data_root.resolve() in args.output.resolve().parents):
        p.error("Refusing output in P0 raw-data root")
    if args.output.resolve() in (
        args.fold_plan.resolve(),args.baseline_report.resolve()):
        p.error("Output may not overwrite source reports")
    source_bytes=args.fold_plan.read_bytes()
    baseline_bytes=args.baseline_report.read_bytes()
    source=json.loads(source_bytes)
    baseline=json.loads(baseline_bytes)
    load_plan(source)
    verify_baseline_report(baseline,source)
    store=CandleStore(args.data_root)
    candles,manifests={},[]
    for sym in SEEN_SYMBOLS:
        print("Reading immutable seen asset",sym,flush=True)
        candles[sym],parts=load_symbol(
            store,sym,months=args.months,verify_hashes=args.verify_hashes)
        manifests.extend(parts)
    report=run_audit(
        candles,source,baseline,fingerprint=manifest_fingerprint(manifests),
        max_folds=args.max_folds)
    report["fold_plan_sha256"]=hashlib.sha256(source_bytes).hexdigest()
    report["baseline_report_sha256"]=hashlib.sha256(baseline_bytes).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print("Saved disagreement audit:",args.output.resolve(),flush=True)
    for h in HORIZONS:
        s=report["summary"][f"H{h}"]["shared_ml_selected"]
        print(
            f"H{h}: eligible_fold_records={s['fold_records_with_rows']} "
            f"selected_rows={s['rows']} "
            f"disagreement_rate={s['disagreement_rate']} "
            f"ML_unique_wins={s['counts']['disagree_ml_correct']} "
            f"reversion_unique_wins={s['counts']['disagree_reversion_correct']} "
            f"pooled_delta_points={s['net_gain_ml_points']}",
            flush=True)


if __name__=="__main__":
    main()
