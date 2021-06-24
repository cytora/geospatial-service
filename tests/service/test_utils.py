import pytest

from service.utils import camel_to_snake_case, headers_is_test, HEADER_PARTNER_USERNAME, HEADER_PARTNER_ID


@pytest.mark.parametrize("text, expected", [
    (
        "thisIsSimple",
        "this_is_simple"
    ),
    (
        "ThisIsAHarderTest",
        "this_is_a_harder_test"
    )
])
def test_camel_to_snake_case(text, expected):
    assert camel_to_snake_case(text) == expected


@pytest.mark.parametrize("headers, expected", [
    (
        {
            HEADER_PARTNER_USERNAME: "cytora-partner",
            HEADER_PARTNER_ID: "123456"
        },
        False
    ),
    (
        {
            HEADER_PARTNER_USERNAME: "cytora-partner"
        },
        False
    ),
    (
        {},
        True
    ),
    (
        {
            HEADER_PARTNER_USERNAME: None,
            HEADER_PARTNER_ID: None
        },
        True
    ),
    (
        {
            "Not-a-user-header": "text"
        },
        True
    )
])
def test_headers_is_test(headers, expected):
    is_test = headers_is_test(headers)
    assert is_test == expected
