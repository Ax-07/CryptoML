"""P3 vendor capacity planning: no live credentials or future outcomes."""
from datetime import datetime,timezone
from mlpipeline.prospective_protocol import make_registration
from mlpipeline.prospective_vendor_plan import provider_plan

def test_vendor_plan_56d_m5_capacity_and_api_incompatibility():
    plan=provider_plan(make_registration(now=datetime(2026,10,14,tzinfo=timezone.utc)))
    assert plan["expected_m5_batches"]==56*24*12==16128
    assert plan["receipts_if_one_per_batch"]==16128
    assert plan["forecast_rows_per_horizon"]==16128*4
    assert plan["contracts_compatible_as_is"] is False
    assert plan["activation_authorized"] is False
    assert plan["independent_append_only_retention_verified"] is False

def test_vendor_plan_verifies_expected_horizons():
    r=make_registration(now=datetime(2026,10,14,tzinfo=timezone.utc))
    r["protocol"]["horizon_bars"]=[2,12]
    import pytest
    with pytest.raises(ValueError,match="H2/H6"):
        provider_plan(r)
