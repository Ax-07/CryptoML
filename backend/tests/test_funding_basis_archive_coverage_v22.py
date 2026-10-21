"""V22 completely offline 4x21x4 source inventory tests."""
from urllib.error import HTTPError
import pytest
import mlpipeline.funding_basis_archive_coverage_v22 as m


def test_exact_sample_and_full_universes():
    a=m.inventory("sample")
    b=m.inventory("full")
    assert len(a)==48 and len(b)==336
    assert len({(s,mo,k) for s,mo,k,_ in b})==336
    assert all(s in m.SYMBOLS for s,_,_,_ in b)
    assert all(mo in m.EXPECTED_MONTHS for _,mo,_,_ in b)
    assert all("/monthly/" in url and url.endswith(".zip") for *_,url in b)
    with pytest.raises(ValueError):
        m.inventory("other")


def test_no_download_only_head_and_404_no_claim(monkeypatch):
    seen=[]
    class Good:
        status=200
        headers={"Content-Length":"123","Last-Modified":"Mon"}
        def __enter__(self):return self
        def __exit__(self,*args):return False
    def fake(request,*,timeout):
        seen.append((request.get_method(),timeout))
        if "2024-10" in request.full_url:
            raise HTTPError(request.full_url,404,"Not Found",{},None)
        return Good()
    monkeypatch.setattr(m,"urlopen",fake)
    assert m.head("https://data.binance.vision/data/BTCUSDT-2024-10.zip")["status"]=="HEAD_404_NOT_CONFIRMED"
    assert m.head("https://data.binance.vision/data/BTCUSDT-2025-06.zip")["status"]=="HEAD_OK_UNVERIFIED"
    assert seen==[("HEAD",12),("HEAD",12)]


def test_report_reconciles_all_48_and_failures():
    jobs=m.inventory("sample")
    rows=[{"symbol":s,"month":month,"kind":kind,
           "status":"HEAD_OK_UNVERIFIED" if i else "UNVERIFIED",
           "http":200 if i else None,"content_length":None,
           "last_modified":None,"error_type":None}
          for i,(s,month,kind,_) in enumerate(jobs)]
    summary,frame=m.summarize(rows,"sample")
    assert summary["expected"]==48 and summary["head_200"]==47
    assert summary["unverified"]==1
    assert len(frame)==48
    assert sum(x["total"] for x in summary["by_symbol"])==48
    with pytest.raises(ValueError,match="incomplete"):
        m.summarize(rows[1:],"sample")
