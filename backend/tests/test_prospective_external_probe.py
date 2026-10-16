"""P3 external provider probe and fail-closed retrieval tests (synthetic only)."""
import base64
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mlpipeline.prospective_anchor import TrustedAnchor
from mlpipeline.prospective_protocol import _canonical, _iso, _sha


class FakeResponse:
    status = 200
    def __init__(self, raw): self.raw = raw
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, n): return self.raw[:n]


def test_provider_post_and_get_must_match(tmp_path, monkeypatch):
    key=Ed25519PrivateKey.generate()
    pem=key.public_key().public_bytes(Encoding.PEM,PublicFormat.SubjectPublicKeyInfo)
    file=tmp_path/"pub.pem"
    file.write_bytes(pem)
    a=TrustedAnchor(url="https://independent.example/attest",
                    lookup_url_template="https://independent.example/proof/{digest}",
                    public_key_path=file)
    close=datetime(2026,10,16,12,tzinfo=timezone.utc)
    core={"bar_close_utc":_iso(close),"protocol_sha256":"a"*64,
          "activation_sha256":"b"*64}
    digest=_sha(_canonical(core))
    signed={"digest_sha256":digest,"issued_at_utc":_iso(close+timedelta(seconds=18)),
            "service_id":"test-provider"}
    response={"signed":signed,"signature_base64":base64.b64encode(
        key.sign(_canonical(signed))).decode("ascii")}
    calls=[]
    def fake_open(req, timeout):
        calls.append(req)
        return FakeResponse(json.dumps(response).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    assert a.attest(core,close=close)==response
    assert len(calls)==2
    assert calls[0].get_method()=="POST" and calls[1].get_method()=="GET"
    assert digest in calls[1].full_url
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req,timeout: FakeResponse(b"{}") if req.get_method()=="GET"
                        else FakeResponse(json.dumps(response).encode()))
    with pytest.raises(ValueError,match="Malformed"):
        a.attest(core,close=close)


def test_local_or_missing_lookup_endpoint_rejected(tmp_path):
    key=Ed25519PrivateKey.generate()
    p=tmp_path/"pub.pem"
    p.write_bytes(key.public_key().public_bytes(
        Encoding.PEM,PublicFormat.SubjectPublicKeyInfo))
    for url in ("https://localhost/attest","http://independent.example/attest",
                "https://127.0.0.1/attest"):
        with pytest.raises(ValueError):
            TrustedAnchor(url=url,lookup_url_template="https://remote.example/{digest}",
                          public_key_path=p)
    with pytest.raises(ValueError,match="placeholder"):
        TrustedAnchor(url="https://remote.example/attest",
                      lookup_url_template="https://remote.example/read",public_key_path=p)
