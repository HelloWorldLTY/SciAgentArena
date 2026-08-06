"""D3 Temporal Validity and Leakage Control scorer.

Evaluates whether all variables are strictly pre-treatment and no
post-t0 features are used. This is a hard validity gate — any leakage
should score 0.
"""

from __future__ import annotations

import re

from eval.normalizers import normalize_text
from eval.schemas import (
    DimensionResult,
    FieldScore,
    TemporalAnswer,
    TemporalGolden,
    VariablesAnswer,
)


def _check_time_zero(agent_t0: str, golden_t0: str) -> FieldScore:
    """Check that the agent's time_zero matches the golden canonical."""
    if not agent_t0.strip():
        return FieldScore(score=0.0, reason="time_zero is empty")

    # Strip trailing punctuation for comparison
    agent_norm = normalize_text(agent_t0).rstrip(".,;:")
    golden_norm = normalize_text(golden_t0).rstrip(".,;:")

    if golden_norm in agent_norm or agent_norm in golden_norm:
        return FieldScore(score=1.0, reason=f"matches: {agent_t0}")

    # Check if core terms overlap (e.g. "hospital admission" in both)
    agent_words = set(agent_norm.split())
    golden_words = set(golden_norm.split())
    overlap = agent_words & golden_words
    if len(overlap) >= 2 and ("admission" in overlap or "time" in overlap):
        return FieldScore(score=1.0, reason=f"keyword match: {overlap}")

    return FieldScore(score=0.0, reason=f"mismatch: {agent_t0} vs {golden_t0}")


def _check_post_t0_features(
    post_t0: list[str],
    forbidden_patterns: list[str],
) -> FieldScore:
    """Check that post_t0_features_used is empty or contains no forbidden features."""
    if not post_t0:
        return FieldScore(score=1.0, reason="no post-t0 features declared")

    # Any post-t0 features is a hard fail
    return FieldScore(
        score=0.0,
        reason=f"post-t0 features used: {post_t0}",
        disqualified=True,
    )


def _check_variable_time_windows(
    variables: VariablesAnswer | None,
    valid_windows: list[str],
    forbidden_patterns: list[str],
) -> FieldScore:
    """Check all declared confounders have valid pre-treatment time windows."""
    if variables is None:
        return FieldScore(score=0.5, reason="no variables block declared")

    violations = []
    total = 0

    for var in variables.confounders + variables.effect_modifiers:
        total += 1
        tw = var.time_window.strip().lower()
        if tw and tw not in [w.lower() for w in valid_windows]:
            violations.append(f"{var.name}: {var.time_window}")

        # Check variable names/definitions for forbidden post-t0 patterns
        var_text = f"{var.name} {var.definition} {var.source}".lower()
        for pattern in forbidden_patterns:
            if re.search(re.escape(pattern.lower()), var_text):
                violations.append(f"{var.name}: contains forbidden pattern '{pattern}'")

    if total == 0:
        return FieldScore(score=0.0, reason="no confounders or effect modifiers declared")

    if not violations:
        return FieldScore(score=1.0, reason=f"all {total} variables have valid time windows")

    if len(violations) == 1:
        return FieldScore(score=0.5, reason=f"1 time window issue: {violations[0]}")

    return FieldScore(
        score=0.0,
        reason=f"{len(violations)} time window violations: {violations[:3]}",
        disqualified=True,
    )


def _score_temporal_verification(report) -> FieldScore:
    """Score based on post-hoc temporal verification results."""
    if report is None:
        return FieldScore(score=0.5, reason="temporal verification skipped (no data/sources)")

    if not report.results:
        return FieldScore(score=0.5, reason="no covariates verified")

    if not report.any_leakage:
        checked = len(report.results)
        return FieldScore(score=1.0, reason=f"all {checked} covariates verified clean")

    leaked = report.covariates_with_leakage
    # Summarize leakage types
    details = []
    for r in report.results:
        if not r.is_clean:
            parts = []
            if r.n_leaked > 0:
                parts.append(f"no_pre_data={r.n_leaked}")
            if r.n_value_matched_post > 0:
                parts.append(f"value_matches_post={r.n_value_matched_post}")
            details.append(f"{r.covariate}({', '.join(parts)})")

    return FieldScore(
        score=0.0,
        reason=f"temporal leakage detected: {'; '.join(details)}",
        disqualified=True,
    )


D3_WEIGHT = 15


def score_d3(
    agent_temporal: TemporalAnswer,
    agent_variables: VariablesAnswer | None,
    golden_temporal: TemporalGolden,
    estimand_time_zero: str = "",
    verification_report=None,
) -> DimensionResult:
    """Score D3 Temporal Validity dimension.

    This is a hard validity gate — any leakage should result in grade 0.
    Falls back to estimand.time_zero if temporal.time_zero is empty.

    Args:
        verification_report: Optional VerificationReport from TemporalVerifier.
            If provided, adds a data-level leakage check.
    """
    field_scores: dict[str, FieldScore] = {}

    # Time zero — fall back to estimand.time_zero if temporal block is empty
    t0 = agent_temporal.time_zero or estimand_time_zero
    field_scores["time_zero"] = _check_time_zero(
        t0, golden_temporal.time_zero_canonical
    )

    # Post-t0 features (self-reported)
    field_scores["post_t0_features"] = _check_post_t0_features(
        agent_temporal.post_t0_features_used,
        golden_temporal.forbidden_post_t0_features,
    )

    # NOTE: variable_time_windows check (#3) dropped — the self-reported
    # "pre_t0" label doesn't capture the valid window (admission → treatment).
    # The programmatic data_verification check below is the real enforcement.

    # Data-level temporal verification (post-hoc)
    if verification_report is not None:
        field_scores["data_verification"] = _score_temporal_verification(verification_report)

    # Hard gate: any disqualification -> grade 0
    any_disqualified = any(fs.disqualified for fs in field_scores.values())

    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if any_disqualified:
        grade = 0
        notes = "HARD GATE: temporal leakage detected"
    elif mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0
        notes = ""

    return DimensionResult(
        dimension="D3",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D3_WEIGHT, 1),
        field_scores=field_scores,
        hard_gate_failed=any_disqualified,
        notes=notes if any_disqualified else "",
    )
