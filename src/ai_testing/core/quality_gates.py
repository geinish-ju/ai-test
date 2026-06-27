from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_testing.core.reports import Record, add_threshold_check


@dataclass(frozen=True)
class QualityGate:
    gate_id: str
    metric_path: str
    operator: str
    threshold: float
    severity: str = "major"
    message: str = ""


def evaluate_quality_gates(
    subject: Mapping[str, Any],
    gates: Sequence[QualityGate],
) -> list[Record]:
    checks: list[Record] = []
    for gate in gates:
        add_threshold_check(
            checks,
            check_id=gate.gate_id,
            value=_float_value(_path_value(subject, gate.metric_path)),
            threshold=gate.threshold,
            direction=gate.operator,
            severity=gate.severity,
            message=gate.message or f"{gate.metric_path} {gate.operator} {gate.threshold}",
        )
    return checks


def _path_value(subject: Mapping[str, Any], path: str) -> Any:
    current: Any = subject
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
