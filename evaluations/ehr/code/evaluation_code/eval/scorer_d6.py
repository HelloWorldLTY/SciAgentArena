"""D6 Diagnostics and Assumption Checking scorer.

Evaluates whether diagnostics (balance, overlap, missingness) are present
and internally consistent.
"""

from __future__ import annotations

from eval.schemas import (
    DiagnosticsAnswer,
    DiagnosticsGolden,
    DimensionResult,
    FieldScore,
)


def _score_balance_from_report(balance_report, golden: DiagnosticsGolden) -> FieldScore:
    """Score balance using the INDEPENDENTLY recomputed SMDs.

    Agent self-reports are ignored — this function only trusts values computed
    directly from the saved cohort file. If the report is missing (no cohort
    file found in run_dir), return 0.
    """
    if balance_report is None:
        return FieldScore(
            score=0.0,
            reason="balance not verifiable (no cohort file found in run_dir) — agent self-report not trusted",
        )

    if not balance_report.covariates:
        return FieldScore(
            score=0.0,
            reason="cohort file found but no numeric covariates present — cannot verify balance",
        )

    smd_before = balance_report.max_smd_before
    smd_after = balance_report.max_smd_after

    if smd_after is not None:
        if smd_after <= golden.balance_smd_threshold:
            return FieldScore(
                score=1.0,
                reason=f"[verified] post-adjustment max SMD={smd_after:.3f} <= {golden.balance_smd_threshold}",
            )
        if smd_after < smd_before:
            return FieldScore(
                score=0.5,
                reason=f"[verified] SMD improved ({smd_before:.3f} -> {smd_after:.3f}) but still above threshold",
            )
        return FieldScore(
            score=0.0,
            reason=f"[verified] post-adjustment SMD ({smd_after:.3f}) not improved vs before ({smd_before:.3f})",
        )

    # No weights in cohort file — only pre-adjustment balance computable
    if smd_before <= golden.balance_smd_threshold:
        return FieldScore(
            score=1.0,
            reason=f"[verified] pre-adjustment SMD={smd_before:.3f} already <= {golden.balance_smd_threshold}",
        )
    return FieldScore(
        score=0.5,
        reason=f"[verified] only pre-adjustment SMD computable (no weights); max SMD={smd_before:.3f}",
    )


def _score_overlap_from_range(
    overlap_range: tuple[float, float] | None, golden: DiagnosticsGolden
) -> FieldScore:
    """Score overlap using the independently recomputed propensity range.

    Agent self-reports are ignored — only values read from the saved cohort
    file's propensity_score column are trusted. If the file or column is
    missing, return 0.
    """
    if overlap_range is None:
        return FieldScore(
            score=0.0,
            reason="overlap not verifiable (no cohort file or no propensity_score column) — agent self-report not trusted",
        )
    lo, hi = overlap_range
    issues = []
    if lo < golden.overlap_min_propensity_threshold:
        issues.append(f"min propensity {lo:.4f} below threshold")
    if hi > golden.overlap_max_propensity_threshold:
        issues.append(f"max propensity {hi:.4f} above threshold")
    if not issues:
        return FieldScore(
            score=1.0,
            reason=f"[verified] propensity range [{lo:.3f}, {hi:.3f}] within bounds",
        )
    return FieldScore(
        score=0.5,
        reason=f"[verified] overlap partially OK: {issues}",
    )


def _score_missingness(missingness) -> FieldScore:
    """Score missingness diagnostics."""
    # Any reporting is better than none
    if missingness.rows_dropped > 0 or missingness.imputation_used:
        return FieldScore(
            score=1.0,
            reason=f"missingness reported: {missingness.rows_dropped} rows dropped, imputation={missingness.imputation_used}",
        )
    # Zero rows dropped could be genuine or just unreported
    return FieldScore(score=0.5, reason="missingness shows 0 rows dropped (may be genuine or unreported)")


def _score_positivity(positivity_violations: list[str]) -> FieldScore:
    """Score positivity violation reporting."""
    if positivity_violations:
        return FieldScore(
            score=1.0,
            reason=f"{len(positivity_violations)} positivity violations reported (transparent)",
        )
    return FieldScore(score=0.5, reason="no positivity violations reported (either none found or not checked)")


D6_WEIGHT = 15


def score_d6(
    agent_diagnostics: DiagnosticsAnswer,
    golden_diagnostics: DiagnosticsGolden,
    balance_report=None,
    overlap_range: tuple[float, float] | None = None,
) -> DimensionResult:
    """Score D6 Diagnostics dimension.

    Balance and overlap are scored from **independently recomputed** values
    derived from the saved cohort file. Agent self-reports in
    `agent_diagnostics.balance` and `agent_diagnostics.overlap` are ignored
    for scoring purposes (they can still be inspected in the answer JSON for
    debugging). If the cohort file is missing or lacks a propensity_score
    column, the corresponding subscore is 0 — we do not trust unverified
    self-attested diagnostics.

    Missingness and positivity_violations remain scored from the agent's
    self-report since they are not cleanly verifiable from the cohort file
    alone.

    Args:
        agent_diagnostics: The agent's self-reported diagnostics block.
        golden_diagnostics: Golden thresholds (SMD cutoff, propensity bounds).
        balance_report: Independently recomputed BalanceReport. None → balance=0.
        overlap_range: Independently recomputed (min, max) propensity range
            from cohort.propensity_score. None → overlap=0.
    """
    field_scores: dict[str, FieldScore] = {}

    field_scores["balance"] = _score_balance_from_report(balance_report, golden_diagnostics)
    field_scores["overlap"] = _score_overlap_from_range(overlap_range, golden_diagnostics)
    field_scores["missingness"] = _score_missingness(agent_diagnostics.missingness)
    field_scores["positivity"] = _score_positivity(agent_diagnostics.positivity_violations)

    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension="D6",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D6_WEIGHT, 1),
        field_scores=field_scores,
    )
