"""V18 source audit: offline synthetic mocked networking, no P0 data."""
from urllib.error import HTTPError
import pytest
import mlpipeline.funding_basis_source_probe_v18 as m


def test_scope_pinned_to_one_symbol_and_three_months():
    with pytest.raises(ValueError,match="scope"):
        m.archive_samples(symbol="ETHUSDT")
    with pytest.raises(ValueError,match="scope"):
        m.archive_samples(months=("2025-06",))
    assert len(m.KINDS)==4


def test_archive_probe_is_bounded_and_non_ingesting(monkeypatch):
    seen=[]
    def fake_head(url):
        seen.append(url)
        return {"state":"HEAD_OK_NOT_INTEGRITY_VERIFIED","http":200}
    monkeypatch.setattr(m,"_head_status",fake_head)
    samples=m.archive_samples()
    assert len(samples)==len(seen)==12
    assert all("BTCUSDT" in url for url in seen)
    assert all(url.endswith(".zip") for url in seen)
    assert set(s["month"] for s in samples)==set(m.MONTHS)
    assert all(s["state"]=="HEAD_OK_NOT_INTEGRITY_VERIFIED" for s in samples)


def test_http_404_not_false_positive(monkeypatch):
    """Simulate urllib.urlopen's actual (request, timeout=...) interface."""
    seen = []

    def fail(request, *, timeout):
        assert timeout == 12
        seen.append((request.get_method(), request.full_url))
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(m, "urlopen", fail)
    result = m._head_status("https://data.binance.vision/data/test.zip")
    assert result["http"] == 404 and result["state"] == "NOT_CONFIRMED"

    funding = m._funding_sample()
    assert funding["http"] == 404 and funding["state"] == "NOT_CONFIRMED"

    assert [method for method, _ in seen] == ["HEAD", "GET"]
    assert "/fapi/v1/fundingRate?" in seen[1][1]
