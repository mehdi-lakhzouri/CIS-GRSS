"""Unambiguous JSON parsing and canonical base64 for public identity artifacts."""

import base64
import binascii
import json


class SecurityError(ValueError):
    """Safe public error: callers must never attach private key/password content."""


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SecurityError("duplicate JSON member")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SecurityError("non-finite JSON number")


def strict_json(text: str) -> dict:
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise SecurityError("invalid or ambiguous JSON object") from exc
    if not isinstance(result, dict):
        raise SecurityError("expected a JSON object")
    return result


def encode64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode64(value: str, length: int) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SecurityError("invalid base64 encoding") from exc
    if len(decoded) != length or encode64(decoded) != value:
        raise SecurityError("invalid length or noncanonical base64 encoding")
    return decoded
