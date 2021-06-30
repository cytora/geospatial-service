import pytest

#from models.requests import ResolveRequest, unmarshal

'''
@pytest.mark.parametrize("req, expected", [
    (
        dict(
            Query="test",
            caller_uid="123",
            min_confidence=0.9
        ),
        ResolveRequest(
            Query="test",
            caller_uid="123",
            min_confidence=0.9
        )
    ),
    (
        dict(
            Query="ltd",
            caller_uid="UWP_caller",
            min_confidence=0.7
        ),
        ResolveRequest(
            Query="ltd",
            caller_uid="UWP_caller",
            min_confidence=0.7
        )
    )
])
def test_unmarshal_match_address_request(req, expected):
    parsed = unmarshal(req, ResolveRequest)
    assert parsed == expected
'''

