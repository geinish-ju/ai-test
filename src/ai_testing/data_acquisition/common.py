from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

JsonObject = Mapping[str, Any]
Record = dict[str, Any]


@dataclass(frozen=True)
class JsonApiClient:
    base_url: str
    service_name: str
    cookies: Mapping[str, str] | str | None = None
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 30.0

    def get_json(self, path: str) -> Any:
        url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        request = Request(url, headers=self._request_headers(), method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code in {401, 403}:
                raise PermissionError(
                    f"{self.service_name} API requires authenticated browser cookies"
                ) from error
            raise ConnectionError(
                f"{self.service_name} API request failed with HTTP {error.code}"
            ) from error
        except URLError as error:
            raise ConnectionError(f"{self.service_name} API request failed: {error}") from error

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "ai-test-framework/0.1",
            **dict(self.headers or {}),
        }
        cookie_header = cookie_header_value(self.cookies)
        if cookie_header:
            headers["Cookie"] = cookie_header
        return headers


def cookie_header_value(cookies: Mapping[str, str] | str | None) -> str | None:
    if cookies is None:
        return None
    if isinstance(cookies, str):
        return cookies
    return "; ".join(f"{key}={value}" for key, value in cookies.items())


def as_mapping(value: Any) -> JsonObject | None:
    return value if isinstance(value, Mapping) else None


def first_value(payload: JsonObject, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def int_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def date_part(value: Any) -> str | None:
    text = text_value(value)
    if text is None:
        return None
    return text[:10] if len(text) >= 10 else text


def mapping_value(payload: JsonObject, key: str) -> JsonObject | None:
    return as_mapping(payload.get(key))


def mapping_list_value(payload: JsonObject, key: str) -> list[JsonObject]:
    return require_mapping_list(payload.get(key), key)


def require_mapping_list(value: Any, source_name: str) -> list[JsonObject]:
    materialized = materialize_mapping_list(value)
    if materialized is None:
        raise TypeError(f"{source_name} must be a JSON array of objects")
    return materialized


def money_amount(value: Any) -> float | None:
    mapping = as_mapping(value)
    if mapping is None:
        return extract_number(value)
    return extract_number(first_value(mapping, ("amount", "value", "price", "total")))


def money_currency(value: Any) -> str | None:
    mapping = as_mapping(value)
    if mapping is None:
        return None
    currency = first_value(mapping, ("currency", "currencyCode"))
    return text_value(currency)


def compact_text_list(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = text_value(value)
        if text is not None:
            result.append(text)
    return result


def extract_mapping_list(payload: Any, candidate_keys: Sequence[str]) -> list[JsonObject]:
    direct = find_candidate_mapping_list(payload, candidate_keys)
    if direct is not None:
        return direct

    fallback = find_first_mapping_list(payload)
    if fallback is not None:
        return fallback

    raise ValueError(f"Cannot find a list for keys: {', '.join(candidate_keys)}")


def find_candidate_mapping_list(
    payload: Any, candidate_keys: Sequence[str]
) -> list[JsonObject] | None:
    mapping = as_mapping(payload)
    if mapping is None:
        return None

    for key in candidate_keys:
        if key in mapping:
            value = mapping[key]
            materialized = materialize_mapping_list(value)
            if materialized is not None:
                return materialized
            nested_result = find_candidate_mapping_list(value, candidate_keys)
            if nested_result is not None:
                return nested_result

    for value in mapping.values():
        nested_result = find_candidate_mapping_list(value, candidate_keys)
        if nested_result is not None:
            return nested_result

    return None


def find_first_mapping_list(value: Any) -> list[JsonObject] | None:
    materialized = materialize_mapping_list(value)
    if materialized is not None:
        return materialized

    mapping = as_mapping(value)
    if mapping is None:
        return None

    for nested in mapping.values():
        result = find_first_mapping_list(nested)
        if result is not None:
            return result
    return None


def materialize_mapping_list(value: Any) -> list[JsonObject] | None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    if not all(isinstance(item, Mapping) for item in value):
        return None
    return [item for item in value]


def extract_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.replace(",", ".").strip()
        try:
            return float(normalized)
        except ValueError:
            return None
    mapping = as_mapping(value)
    if mapping is not None:
        return extract_number(first_value(mapping, ("amount", "value", "price", "total")))
    return None


def ordered_quantity_or_delivered(
    ordered_quantity: float | None,
    delivered_quantity: float | None,
) -> float | None:
    if ordered_quantity == 0.0 and delivered_quantity is not None:
        return delivered_quantity
    return ordered_quantity


def extract_currency(value: Any) -> str | None:
    mapping = as_mapping(value)
    if mapping is None:
        return None
    currency = first_value(mapping, ("currency", "currencyCode"))
    return None if currency is None else str(currency)


def require_json_object(value: Any, service_name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{service_name} API response must be a JSON object")
    return value


def require_json_list(value: Any, service_name: str) -> list[JsonObject]:
    materialized = materialize_mapping_list(value)
    if materialized is None:
        raise TypeError(f"{service_name} API response must be a JSON array of objects")
    return materialized
