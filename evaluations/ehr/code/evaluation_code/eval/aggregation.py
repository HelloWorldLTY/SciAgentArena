"""Aggregation of per-field scores into dimension grades and overall score."""

from __future__ import annotations

from eval.schemas import DimensionResult, FieldScore

# Dimension weights from the eval spec
DIMENSION_WEIGHTS = {
    "D1": 10,
    "D2": 15,
    "D3": 15,
    "D4": 10,
    "D5": 15,
    "D6": 15,
    "D7": 10,
    "D8": 10,
}

# Thresholds for grade assignment
GRADE_2_THRESHOLD = 0.85
GRADE_1_THRESHOLD = 0.50


def aggregate_dimension(
    dimension: str,
    field_scores: dict[str, FieldScore],
    hard_gate_failed: bool = False,
    notes: str = "",
) -> DimensionResult:
    """Aggregate per-field scores into a dimension result.

    Args:
        dimension: Dimension name (e.g. "D1").
        field_scores: Dict mapping field names to FieldScore objects.
        hard_gate_failed: If True, force grade to 0.
        notes: Optional notes.

    Returns:
        DimensionResult with grade (0/1/2), mean score, and weighted score.
    """
    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    weight = DIMENSION_WEIGHTS.get(dimension, 10)

    if hard_gate_failed:
        grade = 0
    elif mean_score >= GRADE_2_THRESHOLD:
        grade = 2
    elif mean_score >= GRADE_1_THRESHOLD:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension=dimension,
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * weight, 1),
        field_scores=field_scores,
        hard_gate_failed=hard_gate_failed,
        notes=notes,
    )


# Keep backward compat
def aggregate_d1(field_scores: dict[str, FieldScore]) -> DimensionResult:
    """Aggregate per-field scores into a D1 dimension result."""
    return aggregate_dimension("D1", field_scores)


def aggregate_all(
    dimension_results: list[DimensionResult],
) -> dict:
    """Aggregate all dimension results into an overall score.

    Returns:
        Dict with overall_score, max_score, percentage, and per-dimension results.
    """
    total_weighted = sum(dr.weighted_score for dr in dimension_results)
    max_possible = sum(2 * DIMENSION_WEIGHTS.get(dr.dimension, 10) for dr in dimension_results)

    return {
        "overall_score": round(total_weighted, 1),
        "max_score": max_possible,
        "percentage": round(total_weighted / max_possible * 100, 1) if max_possible > 0 else 0.0,
        "dimensions": {
            dr.dimension: {
                "grade": dr.grade,
                "mean_field_score": dr.mean_field_score,
                "weighted_score": dr.weighted_score,
                "hard_gate_failed": dr.hard_gate_failed,
                "notes": dr.notes,
            }
            for dr in dimension_results
        },
    }
