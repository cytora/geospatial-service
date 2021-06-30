# pylint: disable=C0413,E0401,E0611

import pytest

'''
@pytest.mark.parametrize("match, expected", [
    (
        MatchML(
            candidate=AddressML(
                id="1",
                components=None,
                properties={},
                full_address=""
            ),
            prob=0.99,
            model_id="model1"
        ),
        AddressMatch(
            id="1",
            confidence=0.99
        )
    )
])
@pytest.mark.asyncio
async def test_map_match(match, expected):
    res = await matcher.search(match)
    assert pytest.approx(expected.confidence, res.confidence, 0.0001)
    assert res.id == expected.id

'''