"""Fail-closed checks for preregistered H6 prospective-only evaluator."""
import pandas as pd
import pytest

from mlpipeline.h6_forward_candidate_evaluation import (
    PROTOCOL, START, END, SYMBOLS, VARIANTS,
    columns_for_models, validate_manifest, validate_rows, compare,
)


def manifest():
    columns=columns_for_models()
    return {
        "protocol":PROTOCOL,"horizon":6,"coverage_target":0.40,
        "uses_ada":False,"uses_fold_oos":False,"uses_reserved_final90":False,
        "models":{
            variant:{
                "feature_columns":list(columns[variant]),
                "model_sha256":"a"*64,"threshold":0.05,
                "train_class_counts":[50,70,50],"tuning_feasible":True,
                "model_frozen_at_utc":"2026-10-16T12:00:00Z",
                "last_training_label_mature_at_utc":"2026-10-15T00:00:00Z",
                "last_tuning_bar_close_at_utc":"2026-10-16T10:00:00Z",
            } for variant in VARIANTS
        },
    }


def minimal_rows():
    t="2026-10-17T00:05:00Z"
    forecasts=[]
    labels=[]
    for symbol in SYMBOLS:
        labels.append({
            "symbol":symbol,"bar_close_at":t,
            "label_mature_at":"2026-10-17T00:35:00Z",
            "target_h6":2,
        })
        for variant in VARIANTS:
            forecasts.append({
                "symbol":symbol,"bar_close_at":t,
                "predicted_at_utc":"2026-10-17T00:05:00Z",
                "variant":variant,"p_down":0.2,
                "p_neutral":0.2,"p_up":0.6,
            })
    return pd.DataFrame(forecasts),pd.DataFrame(labels)


def test_exact_30_25_column_freeze():
    cols=columns_for_models()
    assert len(cols["baseline_30"])==30
    assert len(cols["no_candle_shape_25"])==25
    assert "upper_wick_relative" not in cols["no_candle_shape_25"]


def test_manifest_rejects_changed_candidate_or_future_freeze():
    m=manifest()
    validate_manifest(m)
    m["models"]["no_candle_shape_25"]["feature_columns"].append("upper_wick_relative")
    with pytest.raises(ValueError,match="column mismatch"):
        validate_manifest(m)
    m=manifest()
    m["models"]["baseline_30"]["model_frozen_at_utc"]=START.isoformat()
    with pytest.raises(ValueError,match="precede"):
        validate_manifest(m)


@pytest.mark.parametrize("key,value",[
    ("uses_ada",True),("uses_fold_oos",True),("uses_reserved_final90",True),
    ("coverage_target",0.3),
])
def test_manifest_rejects_sealed_data_or_changed_target(key,value):
    m=manifest()
    m[key]=value
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_no_fresh_data_yet_and_future_label_maturity_are_fail_closed():
    f,l=minimal_rows()
    with pytest.raises(ValueError,match="not yet mature"):
        validate_rows(f,l,as_of_utc="2026-10-18T00:00:00Z")
    with pytest.raises(ValueError,match="Insufficient complete"):
        validate_rows(f,l,as_of_utc="2026-11-14T01:00:00Z")


def test_mismatched_rows_fail_even_when_future_date_arrives():
    f,l=minimal_rows()
    f=f.loc[~((f.symbol=="XRPUSDT")&(f.variant=="no_candle_shape_25"))]
    with pytest.raises(ValueError):
        validate_rows(f,l,as_of_utc=END+pd.Timedelta(days=1))


def test_prospective_cutoff_rejects_historical_label():
    f,l=minimal_rows()
    l.loc[0,"bar_close_at"]="2026-10-16T00:05:00Z"
    with pytest.raises(ValueError,match="outside locked prospective"):
        validate_rows(f,l,as_of_utc=END+pd.Timedelta(days=1))


def test_invalid_probability_and_post_maturity_predictions_fail():
    f,l=minimal_rows()
    f.loc[0,"p_up"]=0.9
    with pytest.raises(ValueError):
        validate_rows(f,l,as_of_utc=END+pd.Timedelta(days=1))
    f,l=minimal_rows()
    f.loc[0,"predicted_at_utc"]="2026-10-17T00:35:00Z"
    with pytest.raises(ValueError):
        validate_rows(f,l,as_of_utc=END+pd.Timedelta(days=1))


def test_scorer_never_runs_before_validated_full_window():
    f,l=minimal_rows()
    with pytest.raises(ValueError):
        compare(f,l,manifest(),as_of_utc="2026-10-16T00:00:00Z")
