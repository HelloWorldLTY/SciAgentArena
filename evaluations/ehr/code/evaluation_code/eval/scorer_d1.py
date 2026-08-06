"""D1 Estimand Specification scoring logic."""

from __future__ import annotations

import difflib

from eval.llm_normalizer import extract_components
from eval.normalizers import check_disqualifiers, normalize_text, parse_iso_duration_to_seconds
from eval.schemas import EstimandAnswer, EstimandGolden, FieldScore


def score_categorical(agent_value: str, golden: dict) -> FieldScore:
    """Score a categorical field (e.g., effect_type) by exact match with alias resolution."""
    canonical = golden["canonical"].strip().lower()
    allowed = [v.lower() for v in golden.get("allowed_values", [])]
    aliases = [a.lower() for a in golden.get("aliases", [])]

    agent_normalized = agent_value.strip().lower()

    # Check exact match to canonical
    if agent_normalized == canonical:
        return FieldScore(score=1.0, reason=f"exact match: {agent_value}")

    # Check alias match
    if agent_normalized in aliases:
        return FieldScore(score=1.0, reason=f"alias match: {agent_value} -> {golden['canonical']}")

    # Check if it's a valid value but wrong
    if agent_normalized in allowed:
        return FieldScore(
            score=0.0,
            reason=f"valid value but incorrect: expected {golden['canonical']}, got {agent_value}",
        )

    return FieldScore(score=0.0, reason=f"invalid or missing value: {agent_value}")


def score_iso_duration(agent_value: str, golden: dict) -> FieldScore:
    """Score an ISO duration field by parsing both to seconds."""
    canonical = golden["canonical"]
    aliases = golden.get("aliases", [])

    canonical_seconds = parse_iso_duration_to_seconds(canonical)
    agent_seconds = parse_iso_duration_to_seconds(agent_value)

    if canonical_seconds is None:
        return FieldScore(score=0.0, reason=f"cannot parse golden canonical: {canonical}")

    if agent_seconds is None:
        # Check if it matches an alias string directly
        agent_lower = agent_value.strip().lower()
        for alias in aliases:
            if agent_lower == alias.lower():
                alias_seconds = parse_iso_duration_to_seconds(alias)
                if alias_seconds == canonical_seconds:
                    return FieldScore(score=1.0, reason=f"alias match: {agent_value} = {canonical}")
        return FieldScore(score=0.0, reason=f"cannot parse agent duration: {agent_value}")

    if agent_seconds == canonical_seconds:
        return FieldScore(score=1.0, reason=f"duration match: {agent_value} = {canonical}")

    return FieldScore(score=0.0, reason=f"duration mismatch: {agent_value} ({agent_seconds}s) != {canonical} ({canonical_seconds}s)")


def score_alias_match(agent_value: str, golden: dict) -> FieldScore:
    """Score a semi-structured field (e.g., time_zero) by alias matching."""
    canonical = golden["canonical"]
    aliases = golden.get("aliases", [])
    disqualifying_values = golden.get("disqualifying_values", [])

    agent_raw_lower = agent_value.lower().strip()
    agent_normalized = normalize_text(agent_value)

    # Check disqualifiers first (check both raw and expanded forms)
    for dq in disqualifying_values:
        dq_lower = dq.lower()
        dq_expanded = normalize_text(dq)
        if dq_lower in agent_raw_lower or dq_lower in agent_normalized or dq_expanded in agent_normalized:
            return FieldScore(score=0.0, disqualified=True, reason=f"disqualified by: {dq}")

    # Check exact match to canonical
    canonical_normalized = normalize_text(canonical)
    if agent_normalized == canonical_normalized:
        return FieldScore(score=1.0, reason=f"exact match to canonical")

    # Check alias matches using fuzzy matching
    all_candidates = [canonical] + aliases
    for candidate in all_candidates:
        candidate_normalized = normalize_text(candidate)
        ratio = difflib.SequenceMatcher(None, agent_normalized, candidate_normalized).ratio()
        if ratio >= 0.85:
            return FieldScore(score=1.0, reason=f"alias match (similarity={ratio:.2f}): {agent_value} ≈ {candidate}")

    # Check if agent text contains any candidate as substring or vice versa
    for candidate in all_candidates:
        candidate_normalized = normalize_text(candidate)
        if candidate_normalized in agent_normalized or agent_normalized in candidate_normalized:
            return FieldScore(score=1.0, reason=f"substring match: {agent_value} ~ {candidate}")

    return FieldScore(score=0.5, reason=f"no alias match found for: {agent_value}")


def score_freetext(
    agent_value: str,
    field_name: str,
    golden: dict,
    use_cache: bool = True,
) -> FieldScore:
    """Score a free-text field using LLM normalization + deterministic component matching."""
    if not agent_value or not agent_value.strip():
        return FieldScore(score=0.0, reason="field is empty or missing")

    required_components = golden.get("required_components", [])
    disqualifying_components = golden.get("disqualifying_components", [])

    # Check disqualifiers via regex (no LLM needed)
    disqualifier_hit = check_disqualifiers(agent_value, disqualifying_components)
    if disqualifier_hit:
        return FieldScore(
            score=0.0,
            disqualified=True,
            reason=f"disqualified by pattern: {disqualifier_hit}",
        )

    if not required_components:
        # No components to check — if we got here, it's not disqualified
        return FieldScore(score=1.0, reason="no required components to check")

    # Use LLM to extract components
    components = extract_components(
        agent_text=agent_value,
        field_name=field_name,
        required_components=required_components,
        use_cache=use_cache,
    )

    n_total = len(required_components)
    n_matched = sum(1 for v in components.values() if v)
    ratio = n_matched / n_total

    if ratio == 1.0:
        score = 1.0
        reason = "all components matched"
    elif ratio >= 0.6:
        missing = [k for k, v in components.items() if not v]
        score = 0.5
        reason = f"{n_matched}/{n_total} components matched ({ratio:.0%}), missing: {missing}"
    else:
        missing = [k for k, v in components.items() if not v]
        score = 0.0
        reason = f"only {n_matched}/{n_total} components matched ({ratio:.0%}), missing: {missing}"

    return FieldScore(score=score, components=components, reason=reason)


# Field name -> scoring function dispatch
_FIELD_SCORERS = {
    "effect_type": "categorical",
    "time_horizon": "iso_duration",
    "time_zero": "alias_match",
    "population": "freetext",
    "treatment": "freetext",
    "comparator": "freetext",
    "outcome": "freetext",
}


def score_d1(
    agent_estimand: EstimandAnswer,
    golden_estimand: EstimandGolden,
    use_cache: bool = True,
) -> dict[str, FieldScore]:
    """Score all D1 fields and return per-field results."""
    field_scores: dict[str, FieldScore] = {}

    for field_name, scorer_type in _FIELD_SCORERS.items():
        agent_value = getattr(agent_estimand, field_name)
        golden_spec = getattr(golden_estimand, field_name).model_dump()

        if scorer_type == "categorical":
            field_scores[field_name] = score_categorical(agent_value, golden_spec)
        elif scorer_type == "iso_duration":
            field_scores[field_name] = score_iso_duration(agent_value, golden_spec)
        elif scorer_type == "alias_match":
            field_scores[field_name] = score_alias_match(agent_value, golden_spec)
        elif scorer_type == "freetext":
            field_scores[field_name] = score_freetext(
                agent_value, field_name, golden_spec, use_cache=use_cache
            )

    return field_scores
