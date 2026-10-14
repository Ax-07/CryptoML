"""P3 external timestamp receipt verification for live forecast batches.

Requires an operator-controlled independent HTTPS service signing digest + time
using Ed25519 and retaining that digest append-only. A signature alone proves
the signer attested to a time, not that the server retained evidence.
No default endpoint and no locally minted 'trusted' timestamps.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_protocol import _canonical, _date, _sha

MAX_ANCHOR_LATENCY_SECONDS = 270


def verify_receipt(
    receipt: dict, core: dict, public_key_pem: bytes, *,
    close: datetime,
) -> dict:
    """Verify signer, exact forecast digest and independently asserted UTC time."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(receipt, dict) or set(receipt) != {"signed", "signature_base64"}:
        raise ValueError("Malformed signed external anchor receipt")
    signed = receipt["signed"]
    if not isinstance(signed, dict) or set(signed) != {
        "digest_sha256", "issued_at_utc", "service_id"
    }:
        raise ValueError("Malformed signed external anchor payload")
    if not isinstance(signed["service_id"], str) or not signed["service_id"]:
        raise ValueError("External anchor service identifier missing")
    if signed["digest_sha256"] != _sha(_canonical(core)):
        raise ValueError("External anchor digest does not match forecast")
    issued = _date(signed["issued_at_utc"])
    when = _date(core["bar_close_utc"])
    if when != close.astimezone(timezone.utc):
        raise ValueError("External receipt anchors wrong candle")
    if not 0 <= (issued - when).total_seconds() < MAX_ANCHOR_LATENCY_SECONDS:
        raise ValueError("External receipt outside strict M5 close window")
    try:
        signature = base64.b64decode(receipt["signature_base64"], validate=True)
        key = load_pem_public_key(public_key_pem)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Trusted anchor key must be Ed25519")
        key.verify(signature, _canonical(signed))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError("Invalid external anchor signature") from exc
    return signed


class TrustedAnchor:
    """Operator-configured external HTTPS signer; no local/fallback signing."""

    def __init__(self, *, url: str, public_key_path: Path):
        host = urllib.parse.urlsplit(url)
        if (host.scheme != "https" or not host.hostname or host.username
                or host.password or host.fragment or
                host.hostname.lower() in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("HTTPS independent remote timestamp service required")
        self.url = url
        self.public_key_pem = public_key_path.read_bytes()
        # Key type validated up front before any API call.
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = load_pem_public_key(self.public_key_pem)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Trusted anchor key must be Ed25519")

    def attest(self, core: dict, *, close: datetime) -> dict:
        request_payload = {
            "digest_sha256": _sha(_canonical(core)),
            "bar_close_utc": core["bar_close_utc"],
            "protocol_sha256": core["protocol_sha256"],
            "activation_sha256": core["activation_sha256"],
        }
        req = urllib.request.Request(
            self.url,
            data=_canonical(request_payload),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status != 200:
                raise ValueError("External timestamp provider did not accept evidence")
            raw = response.read(16_384)
        receipt = json.loads(raw)
        self.verify(receipt, core, close=close)
        return receipt

    def verify(self, receipt: dict, core: dict, *, close: datetime) -> dict:
        return verify_receipt(receipt, core, self.public_key_pem, close=close)
