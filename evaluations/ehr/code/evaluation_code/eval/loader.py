"""Load and validate answer.json and golden.json files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from eval.schemas import AnswerSchema, GoldenSchema


class ValidationGateError(Exception):
    """Raised when a hard validity gate fails."""

    def __init__(self, gate: str, message: str):
        self.gate = gate
        super().__init__(f"Hard gate [{gate}] failed: {message}")


def load_answer(path: str | Path) -> AnswerSchema:
    """Load and validate an agent's answer.json.

    Raises:
        ValidationGateError: If the file fails hard validity gates.
    """
    path = Path(path)

    # Gate 1: file exists
    if not path.exists():
        raise ValidationGateError("file_exists", f"answer.json not found at {path}")

    # Gate 2: valid JSON
    try:
        with open(path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationGateError("valid_json", f"Invalid JSON: {e}")

    # Gate 3: required top-level fields
    if "estimand" not in raw:
        raise ValidationGateError("required_fields", "'estimand' key missing from answer.json")

    # Validate against Pydantic model
    try:
        answer = AnswerSchema.model_validate(raw)
    except ValidationError as e:
        raise ValidationGateError("schema_validation", str(e))

    # Gate: all estimand subfields are non-empty
    estimand = answer.estimand
    for field_name in ["population", "treatment", "comparator", "outcome", "time_zero", "time_horizon", "effect_type"]:
        value = getattr(estimand, field_name)
        if not value or not value.strip():
            raise ValidationGateError("non_empty_fields", f"estimand.{field_name} is empty")

    return answer


def load_golden(path: str | Path) -> GoldenSchema:
    """Load and validate a task's golden.json.

    Raises:
        ValidationGateError: If the file is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise ValidationGateError("file_exists", f"golden.json not found at {path}")

    try:
        with open(path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationGateError("valid_json", f"Invalid JSON: {e}")

    try:
        golden = GoldenSchema.model_validate(raw)
    except ValidationError as e:
        raise ValidationGateError("schema_validation", str(e))

    return golden
