HEADER_PARTNER_USERNAME = 'X-Partner-Username'
HEADER_PARTNER_ID = 'X-Partner-Id'
HEADER_TRACE_ID = 'X-Trace-Id'


def headers_is_test(headers) -> bool:
    return headers.get(HEADER_PARTNER_ID) is None and headers.get(HEADER_PARTNER_USERNAME) is None


def camel_to_snake_case(name):
    return ''.join(
        c if c.islower() else '_' + c.lower()
        for c in name
    ).strip('_')
