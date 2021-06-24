# pylint: skip-file
"""Business Intelligence integration"""
from typing import Optional, Any, Dict

from py_platform_utils.logging import Log
from pydantic import BaseModel

from generated.services.universal_resolver.v1_0 import bi_pb2
from models.requests import ResolveRequest
from models.responses import MatchAddressResponse, Error, AddressMatch


def generate_bi_message(request: MatchAddressRequest,
                        response: MatchAddressResponse,
                        error: Error = None,
                        log: Log = None) -> Optional[bi_pb2.Report]:
    """Construct a BI message"""
    try:
        if error is not None:
            report = bi_pb2.Report(
                error=parse_bi_object(error)
            )
            return report

        request_data = parse_bi_object(request)
        response_data = parse_bi_object(response)

        report = bi_pb2.Report(
            request=request_data,
            response=response_data
        )

        return report

    except Exception as e:
        log.error('error while constructing bi message', error=e)

    return None


def parse_bi_object(obj: BaseModel) -> Any:
    if isinstance(obj, Address):
        return _parse_address(obj)

    if isinstance(obj, MatchAddressRequest):
        return _parse_match_address_request(obj)

    if isinstance(obj, AddressMatch):
        return _parse_address_match(obj)

    if isinstance(obj, MatchAddressResponse):
        return _parse_match_address_response(obj)

    if isinstance(obj, Error):
        return _parse_error(obj)

    return None


# request
def _parse_properties(properties: Dict[str, Any]) -> bi_pb2.Properties:
    if not properties:
        return None

    return bi_pb2.Properties(**properties)


def _parse_address(address: Address) -> bi_pb2.Address:
    if not address:
        return None

    return bi_pb2.Address(
        id=address.id,
        text=address.text,
        buildingName=address.buildingName,
        buildingNumber=address.buildingNumber,
        companyName=address.companyName,
        county=address.county,
        countryCode=address.countryCode,
        geoLocation=address.geoLocation,
        gridReference=address.gridReference,
        level=address.level,
        noise=address.noise,
        postcode=address.postcode,
        streetName=address.streetName,
        suburb=address.suburb,
        townCity=address.townCity,
        unit=address.unit,
        properties=_parse_properties(address.properties)
    )


def _parse_match_address_request(request: MatchAddressRequest) -> bi_pb2.MatchAddressRequest:
    if not request:
        return None

    candidates = None
    if request.candidates:
        candidates = [_parse_address(c) for c in request.candidates]

    return bi_pb2.MatchAddressRequest(
        countryCode=request.countryCode,
        query=_parse_address(request.query),
        candidates=candidates
    )


# response
def _parse_address_match(address_match: AddressMatch) -> bi_pb2.AddressMatch:
    if not address_match:
        return None

    return bi_pb2.AddressMatch(
        id=address_match.id,
        confidence=address_match.confidence
    )


def _parse_match_address_response(response: MatchAddressResponse) -> bi_pb2.MatchAddressResponse:
    if not response:
        return None

    alternatives = None
    if response.alternatives:
        alternatives = [_parse_address_match(a) for a in response.alternatives]

    return bi_pb2.MatchAddressResponse(
        best=_parse_address_match(response.best),
        alternatives=alternatives
    )


def _parse_error(error: Error) -> bi_pb2.Error:
    if not error:
        return None

    return bi_pb2.Error(
        code=error.code,
        description=error.description,
        message=error.message
    )
