from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/defaults.jsonc")
CONFIG_ENV_VAR = "AI_TEST_CONFIG"


def default_config_path() -> str:
    return os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH))


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or default_config_path())
    if not path.exists():
        return {}

    payload = json.loads(_strip_json_comments(path.read_text(encoding="utf-8")))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    index = 0
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < len(text):
            next_char = text[index + 1]
            if next_char == "/":
                result.extend("  ")
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    result.append(" ")
                    index += 1
                continue

            if next_char == "*":
                result.extend("  ")
                index += 2
                while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                    result.append(text[index] if text[index] in "\r\n" else " ")
                    index += 1

                if index + 1 >= len(text):
                    raise ValueError("Unterminated block comment in config file")

                result.extend("  ")
                index += 2
                continue

        result.append(char)
        index += 1

    return "".join(result)


def config_section(config: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)

    return current if isinstance(current, Mapping) else {}


def config_str(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key)
    return value if isinstance(value, str) and value else default


def config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def config_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def config_str_tuple(
    config: Mapping[str, Any],
    key: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return tuple(default)
    if not all(isinstance(item, str) for item in value):
        return tuple(default)
    return tuple(value)
