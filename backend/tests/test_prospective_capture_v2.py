"""P3 dual-policy and fail-closed external attestation synthetic tests."""
import base64
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mlpipeline.prospective_anchor import verify_receipt
from mlpipeline.prospective_live_capture import make_forecasts, _check_forecasts
from mlpipeline.prospective_protocol import _canonical, _iso, _sha
from mlpipeline.prospective_scheduler import due_bar_epoch
from mlpipeline.prospective_journal import verify_chain


def utc(hour=12, minute=0):
    return datetime(2026,10,16,hour,minute,tzinfo=timezone.utc)


def test_dual_policy_recorded_inside_primary_and_outside(monkeypatch):
    from mlpipeline import prospective_live_capture as live
    # Dummy model results: verify only immutable policy routing, no future labels.
    monkeypatch.setattr(live,"_class_probs",lambda m, X: np.tile([0.1,0.2,0.7],(len(X),1)))
    monkeypatch.setattr(live,"decision_and_margin",lambda p,c,alpha:(
        np.full(len(p),2,dtype=int),np.full(len(p),0.15)
    ))
    import mlpipeline.features as features
    rows=pd.DataFrame({
        "symbol":["BTCUSDT","ETHUSDT"],
        "bar_close_at":[pd.Timestamp("2026-10-16T13:00:00Z"),
                        pd.Timestamp("2026-10-16T19:00:00Z")],
        **{col:[0.1,0.1] for col in features.FEATURE_COLUMNS},
    })
    packs={h:{"estimator":object(),"train_prior_counts":[2,3,2]} for h in ("H2","H6")}
    activation={"threshold_policy":{"thresholds":{
        "H2":{"12-17_UTC":0.2,"all_day_UTC":0.1},
        "H6":{"12-17_UTC":0.1,"all_day_UTC":0.2},
    }}}
    got=make_forecasts(rows,packs,activation)
    assert got["H2"]["BTCUSDT"]["policies"]["12-17_UTC"]["selected_direction"] is None
    assert got["H2"]["BTCUSDT"]["policies"]["all_day_UTC"]["selected_direction"]==2
    assert got["H6"]["BTCUSDT"]["policies"]["12-17_UTC"]["selected_direction"]==2
    assert got["H6"]["BTCUSDT"]["policies"]["all_day_UTC"]["selected_direction"] is None
    assert got["H2"]["ETHUSDT"]["policies"]["12-17_UTC"]["in_primary_utc_cohort"] is False


def test_missing_policy_fails_closed():
    forecasts={h:{sym:{"policies":{"all_day_UTC":{}}}
        for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
        for h in ("H2","H6")}
    with pytest.raises(ValueError,match="Both"):
        _check_forecasts(forecasts)


def test_scheduler_never_backdates_and_skips_early_and_late():
    assert due_bar_epoch(utc(12,0)) is None
    assert due_bar_epoch(utc(12,0)+timedelta(seconds=15)) == int(utc(12,0).timestamp())
    assert due_bar_epoch(utc(12,4)+timedelta(seconds=31)) is None
    with pytest.raises(ValueError):
        due_bar_epoch(datetime(2026,10,16,12))


def test_signed_receipt_rejects_wrong_digest_and_late_clock():
    cryptography=pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    key=Ed25519PrivateKey.generate()
    pub=key.public_key().public_bytes(Encoding.PEM,PublicFormat.SubjectPublicKeyInfo)
    core={"bar_close_utc":_iso(utc()),"protocol_sha256":"a"*64,
          "activation_sha256":"b"*64}
    def receipt(digest, time):
        signed={"digest_sha256":digest,"issued_at_utc":_iso(time),"service_id":"synthetic"}
        return {"signed":signed,"signature_base64":base64.b64encode(
            key.sign(_canonical(signed))
        ).decode("ascii")}
    good=receipt(_sha(_canonical(core)),utc()+timedelta(seconds=20))
    assert verify_receipt(good,core,pub,close=utc())["service_id"]=="synthetic"
    with pytest.raises(ValueError,match="digest"):
        verify_receipt(receipt("0"*64,utc()+timedelta(seconds=20)),core,pub,close=utc())
    with pytest.raises(ValueError,match="outside"):
        verify_receipt(receipt(_sha(_canonical(core)),utc()+timedelta(minutes=5)),
                       core,pub,close=utc())
    bad={**good,"signature_base64":base64.b64encode(b"bad").decode()}
    with pytest.raises(ValueError,match="signature"):
        verify_receipt(bad,core,pub,close=utc())


def test_partial_journal_is_not_silently_repaired(tmp_path):
    path=tmp_path/"predictions.jsonl"
    path.write_bytes(b'{"entry_sha256":"broken"}')
    with pytest.raises(ValueError,match="Incomplete"):
        verify_chain(path)
