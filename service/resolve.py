# pylint: skip-file

try:
    from build import Query
except Exception as e:
    print(e)
    from .build import Query
finally:
    print('local imports')


def format_response_payload(matches):
    response = []
    for match in matches:
        response.append({
            'match': match.match,
            'confidence': match.similarity,
            'entity_type': ['company'],
            'available_ids': {'company_id': match.crn},
            })
    return response


def format_request_payload(input_query: Query):
    request = {
        'client_query_id': input_query.client_query_id,
        'query_string': input_query.query,
        'max_number_of_candidates_to_retrieve': input_query.max_number_returns,
        'min_confidence': input_query.min_confidence,
        'caller': input_query.caller_uid,
    }
    return request
