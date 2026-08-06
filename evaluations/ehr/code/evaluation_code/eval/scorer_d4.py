"""D4 Variable Construction and Modality Use scorer.

Evaluates treatment/outcome variable definitions and confounder coverage
against a priority-tiered golden inventory.
"""

from __future__ import annotations

import re

from eval.normalizers import normalize_text
from eval.schemas import (
    ConfounderSpec,
    DimensionResult,
    FieldScore,
    VariablesAnswer,
    VariablesGolden,
)


def _pattern_match(text: str, pattern: str) -> bool:
    """Check if text matches a pipe-separated pattern."""
    text_lower = text.lower()
    text_norm = normalize_text(text)
    for alt in pattern.split("|"):
        alt = alt.strip().lower()
        if alt in text_lower or alt in text_norm:
            return True
    return False


def _score_variable_def(
    variable,
    name_pattern: str,
    label: str,
) -> FieldScore:
    """Score a treatment or outcome variable definition."""
    if variable is None:
        return FieldScore(score=0.0, reason=f"{label} variable not declared")

    var_text = f"{variable.name} {variable.definition}"
    if _pattern_match(var_text, name_pattern):
        return FieldScore(score=1.0, reason=f"{label} variable matches pattern")

    return FieldScore(score=0.5, reason=f"{label} variable '{variable.name}' may not match expected pattern")


def _match_confounder(
    agent_confounders: list,
    golden_spec: ConfounderSpec,
) -> bool:
    """Check if any agent confounder matches a golden confounder spec."""
    for ac in agent_confounders:
        ac_name = ac.name.lower().strip()
        ac_text = f"{ac.name} {ac.definition} {ac.source}".lower()

        # Check direct name match
        golden_name = golden_spec.name.lower()
        if golden_name in ac_name or ac_name in golden_name:
            return True

        # Check aliases
        for alias in golden_spec.aliases:
            alias_lower = alias.lower()
            if alias_lower in ac_text:
                return True

        # Fuzzy: check normalized forms
        golden_norm = normalize_text(golden_spec.name.replace("_", " "))
        ac_norm = normalize_text(ac.name.replace("_", " "))
        if golden_norm in ac_norm or ac_norm in golden_norm:
            return True

    return False


def _score_confounders(
    agent_confounders: list,
    golden_confounders: dict[str, list[ConfounderSpec]],
) -> tuple[FieldScore, dict[str, FieldScore]]:
    """Score confounder coverage against priority tiers.

    Returns (overall_score, per_tier_scores).
    """
    tier_scores: dict[str, FieldScore] = {}
    tier_weights = {"P1": 3.0, "P2": 2.0, "P3": 1.0}

    total_weighted = 0.0
    matched_weighted = 0.0

    for tier_name, specs in golden_confounders.items():
        if not specs:
            continue

        weight = tier_weights.get(tier_name, 1.0)
        matched = []
        missing = []

        for spec in specs:
            if _match_confounder(agent_confounders, spec):
                matched.append(spec.name)
            else:
                missing.append(spec.name)

        n_total = len(specs)
        n_matched = len(matched)
        ratio = n_matched / n_total if n_total > 0 else 0.0

        total_weighted += weight * n_total
        matched_weighted += weight * n_matched

        if ratio >= 0.85:
            score = 1.0
        elif ratio >= 0.5:
            score = 0.5
        else:
            score = 0.0

        tier_scores[tier_name] = FieldScore(
            score=score,
            reason=f"{n_matched}/{n_total} {tier_name} confounders matched"
            + (f", missing: {missing[:5]}" if missing else ""),
        )

    overall_ratio = matched_weighted / total_weighted if total_weighted > 0 else 0.0

    if overall_ratio >= 0.7:
        overall_score = 1.0
    elif overall_ratio >= 0.4:
        overall_score = 0.5
    else:
        overall_score = 0.0

    overall = FieldScore(
        score=overall_score,
        reason=f"weighted confounder coverage: {overall_ratio:.1%}",
    )

    return overall, tier_scores


D4_WEIGHT = 10


def score_d4(
    agent_variables: VariablesAnswer | None,
    golden_variables: VariablesGolden,
) -> DimensionResult:
    """Score D4 Variable Construction dimension."""
    field_scores: dict[str, FieldScore] = {}

    if agent_variables is None:
        return DimensionResult(
            dimension="D4",
            grade=0,
            mean_field_score=0.0,
            weighted_score=0.0,
            field_scores={"variables": FieldScore(score=0.0, reason="no variables block")},
        )

    # Treatment variable
    field_scores["treatment_var"] = _score_variable_def(
        agent_variables.treatment,
        golden_variables.treatment_name_pattern,
        "treatment",
    )

    # Outcome variable
    field_scores["outcome_var"] = _score_variable_def(
        agent_variables.outcome,
        golden_variables.outcome_name_pattern,
        "outcome",
    )

    # Confounders
    if golden_variables.confounders:
        overall, tier_scores = _score_confounders(
            agent_variables.confounders + agent_variables.effect_modifiers,
            golden_variables.confounders,
        )
        field_scores["confounders_overall"] = overall
        for tier, score in tier_scores.items():
            field_scores[f"confounders_{tier}"] = score

    # Imaging
    if golden_variables.imaging_required:
        if agent_variables.imaging_features_used:
            field_scores["imaging"] = FieldScore(score=1.0, reason="imaging features declared")
        else:
            field_scores["imaging"] = FieldScore(score=0.0, reason="imaging required but not used")

    # Aggregate
    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension="D4",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D4_WEIGHT, 1),
        field_scores=field_scores,
    )
