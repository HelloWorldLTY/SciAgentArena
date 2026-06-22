"""oracle_validity — Scorer for the Chemical Claim Validation category.

Assertion-based scoring with weighted checks and gated evaluation.

New in v2.0 (enum + gating redesign):

1. ``value_in_set`` — value must be exactly equal to one of a list of allowed
   strings/primitives. Used for the ``inconclusive_*`` family check on
   ``conclusion_status``.
2. ``value_in_enum`` — value must be drawn from the shared answer pool at
   ``scorers/data/answer_pool.json``. Prevents typo'd enum strings
   from getting partial credit elsewhere.
3. ``requires`` — optional field on any check. If the named prerequisite check
   failed (or was itself gated out), this check is skipped and scored 0
   against full weight. This implements gating: refusal credit only counts
   if the agent actually computed the numeric prerequisite.

Retained for backward compatibility: ``value_equals``, ``value_in_range``,
``less_than``, ``is_true``, ``contains_any``, ``shape_equals``,
``shape_first_dim``, ``plot_exists``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Answer pool loading (cached)
# ---------------------------------------------------------------------------

# The answer pool (enum vocabulary + ground-truth labels) is scorer-side data,
# stored next to the other ground truth in scorers/data/.
_ANSWER_POOL_CANDIDATES = [
    Path(__file__).resolve().parent / "data" / "answer_pool.json",
]
_answer_pool_cache: dict | None = None


def _resolve_answer_pool_path() -> Path:
    for candidate in _ANSWER_POOL_CANDIDATES:
        if candidate.exists():
            return candidate
    return _ANSWER_POOL_CANDIDATES[0]


def _load_answer_pool() -> dict:
    global _answer_pool_cache
    if _answer_pool_cache is None:
        with open(_resolve_answer_pool_path()) as f:
            _answer_pool_cache = json.load(f)
    return _answer_pool_cache


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------


def _check_value_equals(actual, expected, tolerance: float = 0) -> bool:
    if isinstance(expected, bool):
        return actual is True if expected else actual is False
    if isinstance(expected, str):
        return str(actual).strip() == expected.strip()
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            return False
    return actual == expected


def _check_value_in_range(actual, min_val: float, max_val: float) -> bool:
    try:
        v = float(actual)
    except (TypeError, ValueError):
        return False
    return min_val <= v <= max_val


def _check_less_than(actual, threshold: float) -> bool:
    try:
        return float(actual) < threshold
    except (TypeError, ValueError):
        return False


def _check_value_in_set(actual, allowed: list) -> bool:
    try:
        a = str(actual).strip()
    except Exception:
        return False
    return a in [str(x).strip() for x in allowed]


def _check_value_in_enum(actual, enum_field: str) -> bool:
    pool = _load_answer_pool()
    if enum_field not in pool:
        return False
    allowed = pool[enum_field].get("values", [])
    return _check_value_in_set(actual, allowed)


def _shape_matches(actual_shape, expected_shape) -> bool:
    if len(actual_shape) != len(expected_shape):
        return False
    return all(e == -1 or a == e for a, e in zip(actual_shape, expected_shape))


def _get_variable(inspector: dict, var_name: str):
    info = inspector.get(var_name)
    if info is None:
        return False, None
    if isinstance(info, dict) and info.get("exists") is False:
        return False, None
    return True, info


# ---------------------------------------------------------------------------
# Single-check evaluator
# ---------------------------------------------------------------------------


def _evaluate_check(check: dict, inspector_output: dict) -> tuple[bool, dict]:
    """Evaluate one check against the inspector output.

    Returns (passed, result_details). Does NOT consider gating — that is
    handled by the two-pass driver in ``score_validity_task``.
    """
    assertion = check.get("assertion", "")
    result: dict = {"assertion": assertion, "pass": False}

    if assertion == "plot_exists":
        ok = bool(inspector_output.get("__plot_exists__", False))
        result["pass"] = ok
        result["actual"] = inspector_output.get("__plot_exists__")
        return ok, result

    var_name = check.get("variable", "")
    exists, info = _get_variable(inspector_output, var_name)
    if not exists:
        result["error"] = f"Variable '{var_name}' not found"
        return False, result

    if isinstance(info, dict):
        val = info.get("value")
        shape = info.get("shape")
    else:
        val = info
        shape = None

    if assertion == "shape_equals":
        expected_shape = check["expected_shape"]
        ok = shape is not None and _shape_matches(shape, expected_shape)
        result["pass"] = ok
        result["expected_shape"] = expected_shape
        result["actual_shape"] = shape

    elif assertion == "shape_first_dim":
        expected_n = check["expected_n"]
        ok = shape is not None and len(shape) > 0 and shape[0] == expected_n
        result["pass"] = ok
        result["expected_n"] = expected_n
        result["actual_shape"] = shape

    elif assertion == "value_equals":
        expected = check["expected"]
        tolerance = check.get("tolerance", 0)
        ok = _check_value_equals(val, expected, tolerance)
        result["pass"] = ok
        result["expected"] = expected
        result["actual"] = val

    elif assertion == "value_in_range":
        min_v = check["min"]
        max_v = check["max"]
        ok = _check_value_in_range(val, min_v, max_v)
        result["pass"] = ok
        result["min"] = min_v
        result["max"] = max_v
        result["actual"] = val

    elif assertion == "less_than":
        threshold = check["threshold"]
        ok = _check_less_than(val, threshold)
        result["pass"] = ok
        result["threshold"] = threshold
        result["actual"] = val

    elif assertion == "is_true":
        ok = bool(val) is True
        result["pass"] = ok
        result["actual"] = val

    elif assertion == "contains_any":
        keywords = check.get("keywords", [])
        actual_str = str(val).lower() if val is not None else ""
        matched = [kw for kw in keywords if kw.lower() in actual_str]
        ok = len(matched) > 0
        result["pass"] = ok
        result["keywords"] = keywords
        result["matched"] = matched
        result["actual_snippet"] = actual_str[:300]

    elif assertion == "value_in_set":
        allowed = check.get("allowed", [])
        ok = _check_value_in_set(val, allowed)
        result["pass"] = ok
        result["allowed"] = allowed
        result["actual"] = val

    elif assertion == "value_in_enum":
        enum_field = check.get("enum_field", "")
        ok = _check_value_in_enum(val, enum_field)
        pool = _load_answer_pool()
        allowed = pool.get(enum_field, {}).get("values", [])
        result["pass"] = ok
        result["enum_field"] = enum_field
        result["allowed"] = allowed
        result["actual"] = val

    elif assertion == "set_equals":
        expected = set(check.get("expected", []))
        try:
            actual_set = set(val) if isinstance(val, (list, tuple, set)) else set()
        except TypeError:
            actual_set = set()
        ok = expected == actual_set
        result["pass"] = ok
        result["expected"] = sorted(expected)
        result["actual"] = sorted(actual_set) if actual_set else []

    elif assertion == "list_jaccard_min":
        expected = set(check.get("expected", []))
        threshold = float(check.get("threshold", 0.5))
        try:
            actual_set = set(val) if isinstance(val, (list, tuple, set)) else set()
        except TypeError:
            actual_set = set()
        if not expected:
            jac = 0.0
        else:
            inter = len(expected & actual_set)
            union = len(expected | actual_set)
            jac = inter / union if union else 0.0
        ok = jac >= threshold
        result["pass"] = ok
        result["threshold"] = threshold
        result["jaccard"] = round(jac, 4)
        result["expected"] = sorted(expected)
        result["actual"] = sorted(actual_set) if actual_set else []

    else:
        result["error"] = f"Unknown assertion type: {assertion}"
        ok = False

    return ok, result


# ---------------------------------------------------------------------------
# Main scorer (two-pass with gating)
# ---------------------------------------------------------------------------


def score_validity_task(inspector_output: dict | None, task: dict) -> dict:
    """Score notebook variables with weighted, gated assertion checks.

    Two-pass evaluation:
        Pass 1 — evaluate every check ignoring ``requires``.
        Pass 2 — any check whose ``requires`` prerequisite did not pass
                 is marked ``skipped_gated`` and contributes 0 to the
                 numerator but full weight to the denominator.
    """
    checks: list[dict] = task.get("scoring_logic", {}).get("checks", [])
    if not checks:
        return {"validity": 1.0, "correctness": 1.0, "details": {}}

    if inspector_output is None:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "No inspector output"},
        }

    # Pass 1 — raw evaluation
    raw_pass: dict[str, bool] = {}
    raw_result: dict[str, dict] = {}
    for i, check in enumerate(checks):
        label = check.get("label", f"check_{i}")
        ok, result = _evaluate_check(check, inspector_output)
        raw_pass[label] = ok
        raw_result[label] = result

    # Pass 2 — apply gating
    weights: list[float] = []
    pass_flags: list[bool] = []
    details: dict = {}
    gating_fail_count = 0

    for i, check in enumerate(checks):
        label = check.get("label", f"check_{i}")
        weight = float(check.get("weight", 1.0))
        requires = check.get("requires")
        result = dict(raw_result[label])

        if requires and not raw_pass.get(requires, False):
            result["gating_status"] = "skipped_gated"
            result["requires"] = requires
            result["pass"] = False
            effective_pass = False
            gating_fail_count += 1
        else:
            result["gating_status"] = "passed" if raw_pass[label] else "failed"
            if requires:
                result["requires"] = requires
            effective_pass = raw_pass[label]

        weights.append(weight)
        pass_flags.append(effective_pass)
        details[label] = result

    total_weight = sum(weights)
    if total_weight > 0:
        weighted_passed = sum(w for w, p in zip(weights, pass_flags) if p)
        correctness = round(weighted_passed / total_weight, 4)
    else:
        correctness = 1.0

    return {
        "validity": 1.0,
        "correctness": correctness,
        "gating_fail_count": gating_fail_count,
        "details": details,
    }
