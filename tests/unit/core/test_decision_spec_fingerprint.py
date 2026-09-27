import datetime
from tracelens.core.decision_spec import DecisionSpec

def test_decision_spec_fingerprint_roundtrip():
    # Test that a spec with a datetime in extra round-trips to the same fingerprint
    spec = DecisionSpec(
        extra={"time": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)}
    )
    fp1 = spec.fingerprint
    
    # Simulate roundtrip by loading from json
    dumped = spec.model_dump(mode='json')
    spec2 = DecisionSpec.model_validate(dumped)
    fp2 = spec2.fingerprint
    
    assert fp1 == fp2
