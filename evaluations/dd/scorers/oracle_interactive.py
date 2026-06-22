"""oracle_interactive — Generic scorer for notebook-variable and vision-JSON tasks.

Two public functions:

- ``score_notebook_variables(inspector_output, task)`` — checks variables
  extracted by the variable inspector cell against declarative assertions
  in ``task["scoring_logic"]["checks"]``.

- ``score_vision_json(agent_stdout, task)`` — compares agent JSON stdout
  against ground-truth checks for vision (image-reading) tasks.

Assertion types
~~~~~~~~~~~~~~~
Each check dict in ``scoring_logic.checks`` contains an ``assertion`` key
and parameters that depend on the assertion type:

=================  =====================================  ====================
Assertion          Parameters                             Logic
=================  =====================================  ====================
shape_equals       variable, expected_shape               shape matches (−1 = wildcard)
shape_first_dim    variable, expected_n                   shape[0] == N
value_equals       variable, expected, tolerance (opt)    exact / approx match
value_in_range     variable, min, max                     min ≤ value ≤ max
less_than          variable, threshold                    value < threshold
is_true            variable                               bool(value) is True
plot_exists        (none)                                 figure has content
=================  =====================================  ====================
"""

from __future__ import annotations

import json
import math

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_value_equals(actual, expected, tolerance: float = 0) -> bool:
    """Compare *actual* to *expected* with optional numeric tolerance."""
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
    """Return True if *actual* is a number in [min_val, max_val]."""
    try:
        v = float(actual)
    except (TypeError, ValueError):
        return False
    return min_val <= v <= max_val


def _check_less_than(actual, threshold: float) -> bool:
    """Return True if *actual* < *threshold*."""
    try:
        return float(actual) < threshold
    except (TypeError, ValueError):
        return False


def _shape_matches(actual_shape: list | tuple, expected_shape: list) -> bool:
    """Compare shapes; −1 in *expected_shape* is a wildcard dimension."""
    if len(actual_shape) != len(expected_shape):
        return False
    return all(e == -1 or a == e for a, e in zip(actual_shape, expected_shape))


def _get_variable(inspector: dict, var_name: str) -> tuple[bool, dict | None]:
    """Look up *var_name* in inspector output.

    Returns ``(exists, info_dict)``.  ``info_dict`` contains ``value``
    and/or ``shape`` depending on the variable type.
    """
    info = inspector.get(var_name)
    if info is None:
        return False, None
    if isinstance(info, dict) and info.get("exists") is False:
        return False, None
    return True, info


# ---------------------------------------------------------------------------
# Notebook-variable scorer
# ---------------------------------------------------------------------------


def score_notebook_variables(inspector_output: dict | None, task: dict) -> dict:
    """Score notebook variables against ``task["scoring_logic"]["checks"]``.

    Parameters
    ----------
    inspector_output : dict or None
        JSON dict emitted by the variable inspector cell.
    task : dict
        Full task dict.

    Returns
    -------
    dict with keys ``validity``, ``correctness``, ``details``.
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

    passed = 0
    total = len(checks)
    details: dict = {}

    for i, check in enumerate(checks):
        assertion = check.get("assertion", "")
        label = check.get("label", f"check_{i}")
        result: dict = {"assertion": assertion, "pass": False}

        # --- plot_exists ---
        if assertion == "plot_exists":
            ok = bool(inspector_output.get("__plot_exists__", False))
            result["pass"] = ok
            result["actual"] = inspector_output.get("__plot_exists__")
            if ok:
                passed += 1
            details[label] = result
            continue

        # All other assertions require a variable
        var_name = check.get("variable", "")
        exists, info = _get_variable(inspector_output, var_name)

        if not exists:
            result["error"] = f"Variable '{var_name}' not found"
            details[label] = result
            continue

        # Resolve the "value" — could be a scalar or nested dict
        if isinstance(info, dict):
            val = info.get("value")
            shape = info.get("shape")
        else:
            val = info
            shape = None

        # --- shape_equals ---
        if assertion == "shape_equals":
            expected_shape = check["expected_shape"]
            ok = shape is not None and _shape_matches(shape, expected_shape)
            result["pass"] = ok
            result["expected_shape"] = expected_shape
            result["actual_shape"] = shape

        # --- shape_first_dim ---
        elif assertion == "shape_first_dim":
            expected_n = check["expected_n"]
            ok = shape is not None and len(shape) > 0 and shape[0] == expected_n
            result["pass"] = ok
            result["expected_n"] = expected_n
            result["actual_shape"] = shape

        # --- value_equals ---
        elif assertion == "value_equals":
            expected = check["expected"]
            tolerance = check.get("tolerance", 0)
            ok = _check_value_equals(val, expected, tolerance)
            result["pass"] = ok
            result["expected"] = expected
            result["actual"] = val

        # --- value_in_range ---
        elif assertion == "value_in_range":
            min_v = check["min"]
            max_v = check["max"]
            ok = _check_value_in_range(val, min_v, max_v)
            result["pass"] = ok
            result["min"] = min_v
            result["max"] = max_v
            result["actual"] = val

        # --- less_than ---
        elif assertion == "less_than":
            threshold = check["threshold"]
            ok = _check_less_than(val, threshold)
            result["pass"] = ok
            result["threshold"] = threshold
            result["actual"] = val

        # --- is_true ---
        elif assertion == "is_true":
            ok = bool(val) is True
            result["pass"] = ok
            result["actual"] = val

        # --- contains_any (string keyword match) ---
        elif assertion == "contains_any":
            keywords = check.get("keywords", [])
            actual_str = str(val).lower() if val is not None else ""
            matched = [kw for kw in keywords if kw.lower() in actual_str]
            ok = len(matched) > 0
            result["pass"] = ok
            result["keywords"] = keywords
            result["matched"] = matched
            result["actual_snippet"] = actual_str[:300]

        # --- set_equals (order-insensitive list/set equality) ---
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

        # --- list_jaccard_min (top-k overlap scoring) ---
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

        if ok:
            passed += 1
        details[label] = result

    correctness = round(passed / total, 4) if total > 0 else 1.0

    return {
        "validity": 1.0,
        "correctness": correctness,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Vision-JSON scorer
# ---------------------------------------------------------------------------


def score_vision_json(agent_stdout: str, task: dict) -> dict:
    """Score agent JSON output for vision (image-reading) tasks.

    Parameters
    ----------
    agent_stdout : str
        Raw stdout from the agent (expected to be JSON).
    task : dict
        Full task dict.

    Returns
    -------
    dict with keys ``validity``, ``correctness``, ``details``.
    """
    checks: list[dict] = task.get("scoring_logic", {}).get("checks", [])

    # Parse agent JSON
    try:
        agent_data = json.loads(agent_stdout.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"parse_error": str(exc)},
        }

    if not isinstance(agent_data, dict):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"parse_error": "Expected a JSON object"},
        }

    if not checks:
        return {"validity": 1.0, "correctness": 1.0, "details": {}}

    passed = 0
    total = len(checks)
    details: dict = {}

    for i, check in enumerate(checks):
        assertion = check.get("assertion", "value_equals")
        key = check.get("key", "")
        label = check.get("label", f"check_{i}")
        result: dict = {"assertion": assertion, "pass": False}

        actual = agent_data.get(key)
        if actual is None and key not in agent_data:
            result["error"] = f"Key '{key}' not found in agent output"
            details[label] = result
            continue

        if assertion == "value_equals":
            expected = check["expected"]
            tolerance = check.get("tolerance", 0)
            ok = _check_value_equals(actual, expected, tolerance)
            result["pass"] = ok
            result["expected"] = expected
            result["actual"] = actual

        elif assertion == "value_in_range":
            min_v = check["min"]
            max_v = check["max"]
            ok = _check_value_in_range(actual, min_v, max_v)
            result["pass"] = ok
            result["min"] = min_v
            result["max"] = max_v
            result["actual"] = actual

        elif assertion == "is_true":
            ok = bool(actual) is True
            result["pass"] = ok
            result["actual"] = actual

        else:
            result["error"] = f"Unknown assertion type: {assertion}"
            ok = False

        if ok:
            passed += 1
        details[label] = result

    correctness = round(passed / total, 4) if total > 0 else 1.0

    return {
        "validity": 1.0,
        "correctness": correctness,
        "details": details,
    }
