"""D1 Estimand scorer using LLM-judge for semantic fields.

Deterministic scoring for trivially checkable fields (effect_type, time_horizon).
LLM-judge with structured rubrics for semantic fields (population, treatment,
comparator, outcome, time_zero).
"""

from __future__ import annotations

from eval.llm_judge import judge_field
from eval.normalizers import parse_iso_duration_to_seconds
from eval.schemas import EstimandAnswer, EstimandGolden, FieldScore


def _score_effect_type(agent_value: str, golden_spec: dict) -> FieldScore:
    """Deterministic categorical match for effect_type."""
    canonical = golden_spec["canonical"].strip().lower()
    aliases = [a.lower() for a in golden_spec.get("aliases", [])]
    agent_norm = agent_value.strip().lower()

    if agent_norm == canonical:
        return FieldScore(score=1.0, reason=f"exact match: {agent_value}")
    if agent_norm in aliases:
        return FieldScore(score=1.0, reason=f"alias match: {agent_value} -> {golden_spec['canonical']}")
    return FieldScore(score=0.0, reason=f"mismatch: expected {golden_spec['canonical']}, got {agent_value}")


def _score_time_horizon(agent_value: str, golden_spec: dict) -> FieldScore:
    """Deterministic duration match for time_horizon."""
    canonical = golden_spec["canonical"]
    aliases = golden_spec.get("aliases", [])

    canonical_seconds = parse_iso_duration_to_seconds(canonical)
    agent_seconds = parse_iso_duration_to_seconds(agent_value)

    if canonical_seconds is None:
        return FieldScore(score=0.0, reason=f"cannot parse golden: {canonical}")

    if agent_seconds is not None and agent_seconds == canonical_seconds:
        return FieldScore(score=1.0, reason=f"duration match: {agent_value} = {canonical}")

    # Check alias strings
    agent_lower = agent_value.strip().lower()
    for alias in aliases:
        if agent_lower == alias.lower():
            alias_seconds = parse_iso_duration_to_seconds(alias)
            if alias_seconds == canonical_seconds:
                return FieldScore(score=1.0, reason=f"alias match: {agent_value} = {canonical}")

    return FieldScore(
        score=0.0,
        reason=f"duration mismatch: {agent_value} != {canonical}",
    )


def _score_with_judge(
    field_name: str,
    agent_value: str,
    golden_spec: dict,
    model: str = "gpt-4o",
    use_cache: bool = True,
) -> FieldScore:
    """Score a semantic field using the LLM judge."""
    canonical = golden_spec["canonical"]
    result = judge_field(
        field_name=field_name,
        golden_value=canonical,
        agent_value=agent_value,
        model=model,
        use_cache=use_cache,
    )
    return FieldScore(score=result["score"], reason=result["reason"])


# Dispatch: field_name -> scorer type
_DETERMINISTIC_FIELDS = {"effect_type", "time_horizon"}
_JUDGE_FIELDS = {"population", "treatment", "comparator", "outcome", "time_zero"}


def score_d1(
    agent_estimand: EstimandAnswer,
    golden_estimand: EstimandGolden,
    model: str = "gpt-4o",
    use_cache: bool = True,
) -> dict[str, FieldScore]:
    """Score all D1 estimand fields.

    Uses deterministic matchers for effect_type and time_horizon.
    Uses LLM-judge with structured rubrics for all other fields.

    Args:
        agent_estimand: The agent's estimand answer.
        golden_estimand: The golden estimand specification.
        model: LLM model for judge calls.
        use_cache: Whether to cache LLM judge results.

    Returns:
        Dict mapping field names to FieldScore objects.
    """
    field_scores: dict[str, FieldScore] = {}

    all_fields = list(_DETERMINISTIC_FIELDS | _JUDGE_FIELDS)

    for field_name in all_fields:
        agent_value = getattr(agent_estimand, field_name)
        golden_spec = getattr(golden_estimand, field_name).model_dump()

        if field_name == "effect_type":
            field_scores[field_name] = _score_effect_type(agent_value, golden_spec)
        elif field_name == "time_horizon":
            field_scores[field_name] = _score_time_horizon(agent_value, golden_spec)
        else:
            field_scores[field_name] = _score_with_judge(
                field_name, agent_value, golden_spec,
                model=model, use_cache=use_cache,
            )

    return field_scores
