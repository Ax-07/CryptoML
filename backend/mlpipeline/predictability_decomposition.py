"""P2 H2/H6: decompose existing 3-class model predictability (dev checks only).

A: P(non-NEUTRAL) from original multiclass probabilities, against training
   movement prior. AP no-skill reference is CHECK prevalence (descriptive).
B: P(UP | non-NEUTRAL) = p_up / (p_up+p_down), assessed ONLY retrospectively
   on true movers. This is an ORACLE-STRATUM DIAGNOSTIC, never a trading gate.
C: repeat A/B in train-quantile volatility regimes using causal volatility_48.

ONE fixed 30-feature LightGBM per fold/horizon. No alternative architecture,
new targets, threshold optimization, or promotion. Source fold plan from
previously inspected P2 checks, 120d train / 7d early / 3d unused / 4d check,
H12 label-maturity purge; no ADA, OOS-test scoring or final-90-day labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS, build_shared_examples
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_matrix
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL = "p2_h2_h6_baseline_predictability_decomposition_v1"
HORIZONS = (2, 6)
REGIMES = ("LOW", "MID", "HIGH")
MIN_REGIME_TRAIN = 100
MIN_REGIME_CHECK = 30


def _binary_scores(y:np.ndarray,p:np.ndarray,train_prior:float)->dict:
    """Binary skill against a STRICTLY train-derived constant prior."""
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or
        not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or
        (p<0).any() or (p>1).any() or
        not np.isfinite(train_prior) or not 0<train_prior<1):
        raise ValueError("Invalid binary outcomes, raw probabilities or training prior")
    model_brier=float(np.mean((p-y)**2))
    prior_brier=float(np.mean((train_prior-y)**2))
    eps=1e-12
    q=np.clip(p,eps,1-eps)
    prior=np.clip(float(train_prior),eps,1-eps)
    model_logloss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q)))
    prior_logloss=float(-np.mean(y*np.log(prior)+(1-y)*np.log1p(-prior)))
    prevalence=float(y.mean())
    both=len(np.unique(y))==2
    ap=float(average_precision_score(y,p)) if y.any() else None
    return {
        "n":int(len(y)),
        "positive":int(y.sum()),
        "observed_positive_fraction":prevalence,
        "train_positive_prior":float(train_prior),
        "average_precision":ap,
        "ap_no_skill_reference_check_prevalence":prevalence if y.any() else None,
        "ap_minus_prevalence_points":100*(ap-prevalence) if ap is not None else None,
        "roc_auc":float(roc_auc_score(y,p)) if both else None,
        "brier":model_brier,
        "train_prior_brier":prior_brier,
        "brier_skill_vs_train_prior":1-model_brier/prior_brier
            if prior_brier>0 else None,
        "logloss":model_logloss,
        "train_prior_logloss":prior_logloss,
        "logloss_skill_vs_train_prior":1-model_logloss/prior_logloss
            if prior_logloss>0 else None,
    }


def stage_scores(target:np.ndarray,probabilities:np.ndarray,
                 *,training_movement_prior:float,
                 training_up_given_movement:float)->dict:
    """Dissect raw 3-class probabilities, NOT prior-corrected decision margins."""
    y=np.asarray(target,dtype=int)
    probs=np.asarray(probabilities,dtype=float)
    if (y.ndim!=1 or probs.shape!=(len(y),3) or not len(y) or
            not np.isin(y,[0,1,2]).all() or not np.isfinite(probs).all()
            or (probs<0).any() or (probs>1).any()
            or not np.allclose(probs.sum(axis=1),1,rtol=0,atol=1e-5)):
        raise ValueError("Expected aligned y and raw 3-class probabilities")
    movement=1-probs[:,1]
    denominator=probs[:,0]+probs[:,2]
    # Rare collapsed probabilities are uninformative; use training-only
    # conditional prior instead of hiding or peeking at future truth.
    conditional_up=np.divide(
        probs[:,2],denominator,out=np.full(len(y),
            float(training_up_given_movement)),where=denominator>1e-12)
    movement=np.clip(movement,0,1)
    conditional_up=np.clip(conditional_up,0,1)
    move_binary=(y!=1).astype(int)
    first=_binary_scores(move_binary,movement,training_movement_prior)
    movers=move_binary.astype(bool)
    if not movers.any():
        second=None
    else:
        truth_up=(y[movers]==2).astype(int)
        second=_binary_scores(
            truth_up,conditional_up[movers],training_up_given_movement)
        predicted_up=conditional_up[movers]>=0.5
        prior_up=training_up_given_movement>=0.5
        accuracy=float(np.mean(predicted_up==truth_up))
        baseline=float(np.mean(prior_up==truth_up))
        second.update({
            "accuracy_oracle_true_movers":accuracy,
            "train_majority_accuracy_on_true_movers":baseline,
            "accuracy_delta_points":100*(accuracy-baseline),
        })
    return {
        "movement_detection":first,
        "direction_given_TRUE_movement_ORACLE":second,
        "true_movers_count":int(movers.sum()),
        "direction_warning":"Truth-conditioned evaluation is retrospective only; "
                            "the movement subset is not known at inference.",
    }


def _prior_from_train(target:np.ndarray)->tuple[float,float]:
    y=np.asarray(target,dtype=int)
    if y.ndim!=1 or len(y)==0 or not np.isin(y,[0,1,2]).all():
        raise ValueError("Invalid training labels")
    n_movers=int(np.count_nonzero(y!=1))
    n_up=int(np.count_nonzero(y==2))
    if not 0<n_movers<len(y) or not 0<n_up<n_movers:
        raise ValueError("Need movement/neutral and both directions in training")
    return n_movers/len(y),n_up/n_movers


def training_regime_thresholds(train:pd.DataFrame)->dict:
    """Volatility breakpoints use ONLY training rows of each asset."""
    if set(train.symbol)!=set(SEEN_SYMBOLS):
        raise ValueError("Expected the four seen assets in training")
    result={}
    for symbol in SEEN_SYMBOLS:
        values=train.loc[train.symbol.eq(symbol),"volatility_48"].to_numpy(dtype=float)
        if len(values)<100 or not np.isfinite(values).all() or (values<0).any():
            raise ValueError(f"{symbol} insufficient finite training volatility")
        q=np.quantile(values,[1/3,2/3])
        if not np.isfinite(q).all():
            raise ValueError("Invalid training volatility quantiles")
        result[symbol]=[float(q[0]),float(q[1])]
    return result


def regime_codes(frame:pd.DataFrame,thresholds:dict)->np.ndarray:
    if set(frame.symbol)-set(SEEN_SYMBOLS):
        raise ValueError("Unknown asset, including ADA")
    codes=np.full(len(frame),-1,dtype=np.int8)
    symbol_col=frame.symbol.to_numpy()
    vol=frame.volatility_48.to_numpy(dtype=float)
    if not np.isfinite(vol).all():
        raise ValueError("Nonfinite regime feature")
    for symbol in SEEN_SYMBOLS:
        index=np.flatnonzero(symbol_col==symbol)
        if len(index):
            q1,q2=thresholds[symbol]
            codes[index]=np.searchsorted((q1,q2),vol[index],side="right")
    if (codes<0).any() or (codes>2).any():
        raise ValueError("Unassigned market regime")
    return codes


def _fit_baseline(train:pd.DataFrame,early:pd.DataFrame,horizon:int,
                  n_estimators:int,seed:int):
    key=f"target_h{horizon}"
    yt=train[key].to_numpy(dtype=int)
    ye=early[key].to_numpy(dtype=int)
    _prior_from_train(yt)
    if set(ye.tolist())!={0,1,2}:
        raise ValueError("Missing early-stop class")
    model=_cpu_model(binary=False,n_estimators=n_estimators,seed=seed)
    model.fit(
        feature_matrix(train),yt,
        eval_set=[(feature_matrix(early),ye)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)],
    )
    return model


def _mean_valid(seq)->float|None:
    values=[float(x) for x in seq if x is not None]
    return float(np.mean(values)) if values else None


def summarize(folds:list[dict])->dict:
    summary={}
    for h in HORIZONS:
        hk=f"H{h}"
        overall=[f["horizons"][hk]["overall"] for f in folds]
        movement=[x["movement_detection"] for x in overall]
        conditional=[x["direction_given_TRUE_movement_ORACLE"] for x in overall
                     if x["direction_given_TRUE_movement_ORACLE"] is not None]
        summary[hk]={
            "folds":len(overall),
            "movement": {
                "mean_ap":_mean_valid(v["average_precision"] for v in movement),
                "mean_ap_excess_over_check_prevalence_points":_mean_valid(
                    v["ap_minus_prevalence_points"] for v in movement),
                "mean_brier_skill_vs_train_prior":_mean_valid(
                    v["brier_skill_vs_train_prior"] for v in movement),
                "mean_roc_auc":_mean_valid(v["roc_auc"] for v in movement),
                "folds_positive_brier_skill":sum(
                    v["brier_skill_vs_train_prior"] is not None and
                    v["brier_skill_vs_train_prior"]>0 for v in movement),
            },
            "direction_oracle": {
                "mean_accuracy":_mean_valid(
                    v["accuracy_oracle_true_movers"] for v in conditional),
                "mean_train_majority_accuracy":_mean_valid(
                    v["train_majority_accuracy_on_true_movers"] for v in conditional),
                "mean_accuracy_delta_points":_mean_valid(
                    v["accuracy_delta_points"] for v in conditional),
                "mean_brier_skill_vs_train_prior":_mean_valid(
                    v["brier_skill_vs_train_prior"] for v in conditional),
                "mean_roc_auc":_mean_valid(v["roc_auc"] for v in conditional),
                "folds_positive_brier_skill":sum(
                    v["brier_skill_vs_train_prior"] is not None and
                    v["brier_skill_vs_train_prior"]>0 for v in conditional),
            },
            "regimes":{},
        }
        for regime in REGIMES:
            for symbol in SEEN_SYMBOLS:
                cells=[
                    f["horizons"][hk]["regimes"][symbol][regime]
                    for f in folds
                ]
                good=[v for v in cells if v["status"]=="OK"]
                stage_a=[v["metrics"]["movement_detection"] for v in good]
                stage_b=[v["metrics"]["direction_given_TRUE_movement_ORACLE"]
                         for v in good
                         if v["metrics"]["direction_given_TRUE_movement_ORACLE"] is not None]
                summary[hk]["regimes"][f"{symbol}/{regime}"]={
                    "valid_folds":len(good),
                    "mean_check_rows":_mean_valid(v["check_rows"] for v in cells),
                    "mean_movement_rate":_mean_valid(
                        v["observed_positive_fraction"] for v in stage_a),
                    "mean_movement_ap_excess_points":_mean_valid(
                        v["ap_minus_prevalence_points"] for v in stage_a),
                    "mean_movement_brier_skill":_mean_valid(
                        v["brier_skill_vs_train_prior"] for v in stage_a),
                    "mean_oracle_direction_accuracy_delta_points":_mean_valid(
                        v["accuracy_delta_points"] for v in stage_b),
                    "mean_oracle_direction_brier_skill":_mean_valid(
                        v["brier_skill_vs_train_prior"] for v in stage_b),
                }
    return summary


def run_audit(candles:dict[str,pd.DataFrame],source:dict,*,
              fingerprint:str,max_folds:int|None=None,
              n_estimators:int=150,seed:int=42)->dict:
    plan=load_plan(source)
    if source.get("dataset_fingerprint_sha256")!=fingerprint:
        raise ValueError("Historical dataset fingerprint mismatch")
    if n_estimators<20 or seed!=42:
        raise ValueError("Prescribed CPU estimator count >=20, seed=42")
    if max_folds is not None:
        if type(max_folds) is not int or not 1<=max_folds<=len(plan):
            raise ValueError("Invalid smoke fold count")
        plan=plan[:max_folds]
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Exactly four seen historical assets; never ADA")
    cutoff=plan[-1]["check_end"]
    # Mandatory: CLIP BEFORE constructing ANY feature or future labels.
    clipped={
        symbol:df.loc[df.bar_close_at.lt(cutoff)].copy()
        for symbol,df in candles.items()
    }
    shared=build_shared_examples(clipped)
    del clipped
    if shared.label_mature_at.ge(cutoff).any():
        raise ValueError("Constructed labels are not mature before last check end")
    t,m=shared.bar_close_at,shared.label_mature_at
    result=[]
    for fold in plan:
        end=fold["check_end"]
        start=fold["check_start"]
        stop=start-pd.Timedelta(days=3)
        val=stop-pd.Timedelta(days=7)
        train_start=val-pd.Timedelta(days=120)
        train=shared.loc[t.ge(train_start)&t.lt(val)&m.lt(val)]
        early=shared.loc[t.ge(val)&t.lt(stop)&m.lt(stop)]
        # Deliberately UNUSED tuning interval [stop,start).
        check=shared.loc[t.ge(start)&t.lt(end)&m.lt(end)]
        if (len(check)!=fold["expected_check_rows"] or
                min(len(train),len(early),len(check))<100 or
                set(check.symbol)!=set(SEEN_SYMBOLS)):
            raise ValueError(f"Mismatch with purged P2 geometry on fold {fold['fold_id']}")
        expected_partition=source["folds"][fold["fold_id"]]["partition_rows"]
        if (len(train)!=expected_partition["train"] or
                len(early)!=expected_partition["early_stopping"]):
            raise ValueError("Train/early geometry differs from original P2 report")
        if not (train.label_mature_at.lt(val).all() and
                early.label_mature_at.lt(stop).all() and
                check.label_mature_at.lt(end).all()):
            raise ValueError("Maturity purge violation")
        thresholds=training_regime_thresholds(train)
        check_regimes=regime_codes(check,thresholds)
        train_regimes=regime_codes(train,thresholds)
        symbols_train=train.symbol.to_numpy()
        symbols_check=check.symbol.to_numpy()
        horizons={}
        for h in HORIZONS:
            hk=f"H{h}"
            target=f"target_h{h}"
            model=_fit_baseline(train,early,h,n_estimators,seed)
            probs=_class_probs(model,feature_matrix(check))
            y_train=train[target].to_numpy(dtype=int)
            y_check=check[target].to_numpy(dtype=int)
            prior_move,prior_up=_prior_from_train(y_train)
            overall=stage_scores(
                y_check,probs,training_movement_prior=prior_move,
                training_up_given_movement=prior_up)
            by_symbol={}
            regimes={}
            for sym in SEEN_SYMBOLS:
                is_train=symbols_train==sym
                is_check=symbols_check==sym
                sym_train_y=y_train[is_train]
                sym_prior=_prior_from_train(sym_train_y)
                by_symbol[sym]=stage_scores(
                    y_check[is_check],probs[is_check],
                    training_movement_prior=sym_prior[0],
                    training_up_given_movement=sym_prior[1])
                regimes[sym]={}
                for idx,name in enumerate(REGIMES):
                    fitmask=is_train & (train_regimes==idx)
                    evalmask=is_check & (check_regimes==idx)
                    fit_n=int(fitmask.sum())
                    check_n=int(evalmask.sum())
                    if fit_n<MIN_REGIME_TRAIN or check_n<MIN_REGIME_CHECK:
                        regimes[sym][name]={
                            "status":"INSUFFICIENT_ROWS",
                            "train_rows":fit_n,"check_rows":check_n,"metrics":None}
                        continue
                    try:
                        move_p,up_p=_prior_from_train(y_train[fitmask])
                    except ValueError:
                        regimes[sym][name]={
                            "status":"INSUFFICIENT_CLASSES",
                            "train_rows":fit_n,"check_rows":check_n,"metrics":None}
                        continue
                    regimes[sym][name]={
                        "status":"OK",
                        "train_rows":fit_n,"check_rows":check_n,
                        "metrics":stage_scores(
                            y_check[evalmask],probs[evalmask],
                            training_movement_prior=move_p,
                            training_up_given_movement=up_p)}
            horizons[hk]={
                "best_iteration":int(model.best_iteration_),
                "train_priors":{"movement":prior_move,"up_given_movement":prior_up},
                "overall":overall,
                "by_symbol":by_symbol,
                "train_volatility_regime_cutoffs":thresholds,
                "regimes":regimes,
            }
            a=overall["movement_detection"]
            b=overall["direction_given_TRUE_movement_ORACLE"]
            print(
                f"fold={fold['fold_id']} {hk} "
                f"movement_AP_excess={a['ap_minus_prevalence_points']:+.3f}pp "
                f"movement_Brier_skill={a['brier_skill_vs_train_prior']:+.5f} "
                f"direction_oracle_accuracy_gain={b['accuracy_delta_points']:+.3f}pp",
                flush=True)
        result.append({
            "fold_id":fold["fold_id"],
            "train_start_utc":train_start.isoformat(),
            "validation_start_utc":val.isoformat(),
            "early_stop_end_exclusive_utc":stop.isoformat(),
            "unused_tuning_end_exclusive_utc":start.isoformat(),
            "check_end_exclusive_utc":end.isoformat(),
            "rows":{"train":len(train),"early_stopping":len(early),
                    "validation_check":len(check)},
            "horizons":horizons,
        })
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "folds_analyzed":len(result),
        "last_allowed_dev_check_end_exclusive_utc":cutoff.isoformat(),
        "label_schema":LABEL_SCHEMA_VERSION,
        "label_threshold_multiplier":0.5,
        "feature_schema":FEATURE_SCHEMA_VERSION,
        "features":list(FEATURE_COLUMNS),
        "model":"same_frozen_reference_hyperparameters_CPU_but_new_diagnostic_fits",
        "n_estimators_max":n_estimators,"seed":seed,
        "horizons":list(HORIZONS),
        "regimes":list(REGIMES),
        "train_only_regime_quantile_breaks":[1/3,2/3],
        "tuning_interval_used":False,
        "ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,
        "final90_scored":False,
        "frozen_P3_or_forward_models_modified":False,
        "limitations":[
            "Fresh re-fits to old dev folds are NOT independent validation.",
            "Movement AP prevalence baseline is check-label descriptive, never fitted.",
            "Conditional direction scores select realized movers in hindsight (oracle stratum).",
            "Volatility regimes are based on observable trailing volatility, but "
              "class-conditional stage B is not deployable without a movement predictor.",
            "No threshold or feature/label selection; no PnL, costs or deployment.",
            "Folds share assets and overlapping training history; fold means are not iid.",
        ],
        "summary":summarize(result),
        "folds":result,
    }


def main()->None:
    p=argparse.ArgumentParser(
        description="P2 CPU H2/H6: movement vs direction skill of existing 30-feature LightGBM")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--fold-plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int,default=None)
    p.add_argument("--n-estimators",type=int,default=150)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24 or a.n_estimators!=150 or a.seed!=42:
        p.error("Fixed P2: months=24 n-estimators=150 seed=42")
    if a.output.exists():
        p.error("Refusing to overwrite an existing report")
    if a.data_root.resolve() in a.output.resolve().parents:
        p.error("Report must not be under immutable source data root")
    source_bytes=a.fold_plan.read_bytes()
    source=json.loads(source_bytes)
    load_plan(source)  # fail on invalid/suspect plan BEFORE reading any candles
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for sym in SEEN_SYMBOLS:
        print("Reading immutable seen asset",sym,flush=True)
        candles[sym],parts=load_symbol(
            store,sym,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(parts)
    report=run_audit(candles,source,
                     fingerprint=manifest_fingerprint(manifests),
                     max_folds=a.max_folds,
                     n_estimators=a.n_estimators,seed=a.seed)
    report["fold_plan_sha256"]=hashlib.sha256(source_bytes).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved P2 movement/direction predictability report:",a.output.resolve(),flush=True)
    for h in HORIZONS:
        s=report["summary"][f"H{h}"]
        print(
            f"H{h}: movement_mean_AP_excess="
            f"{s['movement']['mean_ap_excess_over_check_prevalence_points']:+.3f}pp "
            f"movement_mean_Brier_skill={s['movement']['mean_brier_skill_vs_train_prior']:+.5f} "
            f"direction_oracle_mean_accuracy_delta="
            f"{s['direction_oracle']['mean_accuracy_delta_points']:+.3f}pp "
            f"direction_oracle_mean_Brier_skill="
            f"{s['direction_oracle']['mean_brier_skill_vs_train_prior']:+.5f}",
            flush=True)


if __name__=="__main__":
    main()
