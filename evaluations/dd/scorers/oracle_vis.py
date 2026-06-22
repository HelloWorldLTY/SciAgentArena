"""oracle_vis — Matplotlib figure attribute scoring.

Scores are computed by comparing inspector output (extracted from a hidden
notebook cell) against the ``expected_plot`` specification in the task JSON.

Scorer protocol
~~~~~~~~~~~~~~~
``check_plot_attributes(inspector_output, expected_plot)`` returns::

    {
        "validity":    float,   # 1.0 if inspector produced parseable output
        "correctness": float,   # fraction of checks that passed
        "details":     dict,    # per-check breakdown
    }
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Fuzzy label matching
# ---------------------------------------------------------------------------


def _fuzzy_label_match(actual: str | None, expected: str) -> bool:
    """Case-insensitive, whitespace-normalized comparison of axis labels."""
    if actual is None:
        return False
    norm_actual = re.sub(r"\s+", " ", actual.strip().lower())
    norm_expected = re.sub(r"\s+", " ", expected.strip().lower())
    return norm_actual == norm_expected


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


def check_plot_attributes(inspector_output: dict, task: dict) -> dict:
    """Score inspector results against the task's ``expected_plot`` spec.

    Parameters
    ----------
    inspector_output : dict
        Keys produced by the notebook inspector cell:
        ``x_label``, ``y_label``, ``title``, ``n_collections``,
        ``n_lines``, ``n_patches``, ``has_colorbar``.
    task : dict
        Full task dict; ``task["scoring_logic"]["expected_plot"]`` is used.
    """
    expected: dict = task.get("scoring_logic", {}).get("expected_plot", {})
    details: dict = {}
    checks_passed = 0
    checks_total = 0

    # --- x_label ---
    if "x_label" in expected:
        checks_total += 1
        ok = _fuzzy_label_match(inspector_output.get("x_label"), expected["x_label"])
        details["x_label"] = {
            "expected": expected["x_label"],
            "actual": inspector_output.get("x_label"),
            "pass": ok,
        }
        checks_passed += int(ok)

    # --- y_label ---
    if "y_label" in expected:
        checks_total += 1
        ok = _fuzzy_label_match(inspector_output.get("y_label"), expected["y_label"])
        details["y_label"] = {
            "expected": expected["y_label"],
            "actual": inspector_output.get("y_label"),
            "pass": ok,
        }
        checks_passed += int(ok)

    # --- title ---
    if "title" in expected:
        checks_total += 1
        ok = _fuzzy_label_match(inspector_output.get("title"), expected["title"])
        details["title"] = {
            "expected": expected["title"],
            "actual": inspector_output.get("title"),
            "pass": ok,
        }
        checks_passed += int(ok)

    # --- plot_type → map to element counts ---
    if "plot_type" in expected:
        checks_total += 1
        plot_type = expected["plot_type"]
        if plot_type == "scatter":
            count = inspector_output.get("n_collections", 0)
            ok = count >= 1
        elif plot_type == "line":
            count = inspector_output.get("n_lines", 0)
            ok = count >= 1
        elif plot_type == "bar":
            count = inspector_output.get("n_patches", 0)
            ok = count >= 1
        else:
            ok = False
            count = None
        details["plot_type"] = {
            "expected": plot_type,
            "element_count": count,
            "pass": ok,
        }
        checks_passed += int(ok)

    # --- min_points ---
    if "min_points" in expected:
        checks_total += 1
        # For scatter: total offsets across PathCollections
        n_points = inspector_output.get("n_scatter_points", 0)
        ok = n_points >= expected["min_points"]
        details["min_points"] = {
            "expected": expected["min_points"],
            "actual": n_points,
            "pass": ok,
        }
        checks_passed += int(ok)

    # --- has_colorbar ---
    if "has_colorbar" in expected:
        checks_total += 1
        ok = inspector_output.get("has_colorbar", False) == expected["has_colorbar"]
        details["has_colorbar"] = {
            "expected": expected["has_colorbar"],
            "actual": inspector_output.get("has_colorbar"),
            "pass": ok,
        }
        checks_passed += int(ok)

    correctness = round(checks_passed / checks_total, 4) if checks_total > 0 else 1.0

    return {
        "validity": 1.0,
        "correctness": correctness,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Stub for future image-based scoring
# ---------------------------------------------------------------------------


def check_plot_exists(inspector_output: dict) -> bool:
    """Return True if inspector found a non-empty matplotlib figure."""
    if "__plot_exists__" in inspector_output:
        return bool(inspector_output["__plot_exists__"])
    return (
        inspector_output.get("n_collections", 0) > 0
        or inspector_output.get("n_lines", 0) > 0
        or inspector_output.get("n_patches", 0) > 0
    )


def check_saved_image(image_path: str, task: dict) -> dict:
    """Score a saved PNG/SVG figure against task expectations."""
    raise NotImplementedError("check_saved_image is not yet implemented")
